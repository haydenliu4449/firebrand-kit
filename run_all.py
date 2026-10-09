#!/usr/bin/env python3
"""The whole pipeline as a resumable script. No notebook, no kernel to lose.

    python run_all.py --clips clips/ --work work --mask masks/left_front.json

Runs detect -> tiles -> train -> eval -> tag. **Every stage writes its result
to disk and is skipped if that result already exists**, so a crash, a dropped
connection, a preempted Spot VM or a machine you turned off costs you only the
stage that was running -- never the hours before it. Training goes further and
resumes mid-run from the last finished epoch.

ON GOOGLE CLOUD BATCH this is what every job runs (via batch_entry.py), with
--clips and --work pointing into the bucket mounted at /mnt/disks/share. You do
not normally call it directly there -- `bash scripts/04-submit_job.sh <step>`
does. See the README.

LOCALLY it works exactly the same:

    python run_all.py --clips clips/ --work work --stage detect
    python run_all.py --clips clips/ --work work --stage tiles,train,eval

STAGES

  detect   each clip -> tracks.csv, detections.csv, contact sheets, heatmap
  tiles    verified tracks -> fitted generator -> data/tiles.npz
  train    tiles -> runs/streak/best.pt   (resumes from runs/streak/last.pt)
  eval     model vs classical detector on held-out synthetic -> report.json
  tag      the trained model applied to every real clip -> tags/<clip>/

`--stage` takes one name, a comma list (tiles,train,eval) or `all`.

Between `detect` and `tiles`, review the contact sheets and write the track IDs
that are NOT firebrands into work/clips/<clip>/reject.txt (commas or newlines).

SHARDING. detect and tag work clip by clip, so they can be split across
several machines: `--shard 2/5` processes clips 2, 7, 12, ... of the sorted
list. `--shard auto` reads Batch's BATCH_TASK_INDEX / BATCH_TASK_COUNT, which is
how a multi-task Batch job fans the clips out. The other stages need all clips
at once, so only shard 0 runs them and the rest log a skip.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from firebrand import Config, frames as F, detect as D, track as T, review as R
from firebrand.config import SynthConfig
from firebrand import synth as S, dataset as DS, evaluate as E, gcsio

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}


class _Gen:
    """Re-iterable frame source over a generator factory.

    The pipeline's consumers each want to walk the clip from the start, and a
    plain generator is exhausted after one pass. This re-opens the stream per
    iteration so nothing has to hold the frames.
    """

    def __init__(self, factory):
        self.factory = factory

    def __iter__(self):
        return iter(self.factory())


# ---------------------------------------------------------------------------
# logging: timestamped and flushed, so `tail -f` is actually live
# ---------------------------------------------------------------------------

def log(msg, *, rule=False):
    if rule:
        print("\n" + "=" * 68, flush=True)
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)
    if rule:
        print("=" * 68, flush=True)


class Timer:
    def __init__(self, what):
        self.what = what

    def __enter__(self):
        self.t0 = time.time()
        log(f"{self.what} ...")
        return self

    def __exit__(self, *a):
        log(f"{self.what} done in {time.time() - self.t0:.0f}s")


def progress_printer(total, every_s=20.0):
    """Sparse progress for a log file. A carriage-return bar is useless in
    nohup output -- it just writes one enormous line."""
    state = {"last": 0.0, "t0": time.time()}

    def p(k):
        now = time.time()
        if now - state["last"] < every_s and k + 1 < total:
            return
        state["last"] = now
        el = now - state["t0"]
        rate = (k + 1) / max(el, 1e-6)
        log(f"    frame {k+1}/{total}  {rate:.1f} fps  "
            f"eta {(total - k - 1)/max(rate,1e-6)/60:.1f} min")
    return p


# ---------------------------------------------------------------------------
# stage: detect
# ---------------------------------------------------------------------------

def list_clips(spec):
    if gcsio.is_gcs(spec):
        return sorted(o for o in gcsio.listdir(spec)
                      if Path(o).suffix.lower() in VIDEO_EXTS)
    p = Path(spec)
    if p.is_file():
        return [str(p)]
    if not p.is_dir():
        raise SystemExit(f"clips folder not found: {p}")
    return sorted(str(q) for q in p.iterdir()
                  if q.suffix.lower() in VIDEO_EXTS)


# ---------------------------------------------------------------------------
# sharding: one Batch job, many machines, each taking a slice of the clips
# ---------------------------------------------------------------------------

def parse_shard(spec):
    """'auto' | 'I/N' | None -> (index, count).

    'auto' reads the variables Cloud Batch sets in every task. Outside Batch
    they are absent and this is (0, 1): process everything.
    """
    if not spec:
        return 0, 1
    if spec == "auto":
        return (int(os.environ.get("BATCH_TASK_INDEX", 0)),
                int(os.environ.get("BATCH_TASK_COUNT", 1)))
    i, _, n = spec.partition("/")
    i, n = int(i), int(n)
    if not (n >= 1 and 0 <= i < n):
        raise SystemExit(f"bad --shard {spec!r}: want I/N with 0 <= I < N")
    return i, n


def my_clips(args):
    clips = list_clips(args.clips)
    if not clips:
        raise SystemExit(f"no video files found in {args.clips}")
    i, n = args.shard
    mine = clips[i::n]
    if n > 1:
        log(f"shard {i}/{n}: {len(mine)} of {len(clips)} clip(s): "
            f"{', '.join(Path(c).stem for c in mine) or '(none)'}")
    else:
        log(f"{len(clips)} clip(s) to process")
    return mine


def px_scale(W):
    """How much bigger this clip's pixels-per-metre is than the 1920-wide
    frames the pixel limits were tuned on. Never below 1: at lower resolution
    the limits stay as they were (permissive) rather than tightening."""
    return max(1.0, float(W) / 1920.0)


def clip_config(cfg, W, H, fps):
    """Per-clip detection settings, scaled to this clip's resolution.

    Every limit that is a size in pixels was tuned at 1920 wide. At 3840 wide
    (these cameras) a streak, an area and a per-frame step are 2x, 4x and 2x as
    many pixels, so the upper limits scale with the frame; otherwise the
    fastest and closest embers -- the brightest ones -- are the first thrown
    away. Lower limits are left alone: scaling them would only lose faint ones.
    """
    c = Config()
    c.detect.__dict__.update(cfg.detect.__dict__)
    c.track.__dict__.update(cfg.track.__dict__)
    c.train.__dict__.update(cfg.train.__dict__)
    c.track.fps = float(fps or 15)
    f = px_scale(W)
    c.detect.tophat_ksize = max(5, int(round(15 * W / 1920)) | 1)
    c.detect.max_length = cfg.detect.max_length * f
    c.detect.max_area = int(round(cfg.detect.max_area * f * f))
    c.track.max_step_px = cfg.track.max_step_px * f
    if W * H > 2_500_000 and c.detect.bg_refresh == 1:
        c.detect.bg_refresh = 4
    return c


def roi_for(args, first):
    H, W = first.shape[:2]
    if args.mask:
        return F.mask_from_spec((H, W), F.load_mask_spec(args.mask))
    if not args.no_mask:
        return F.make_roi_mask((H, W), **F.LEFT_FRONT_OVERLAYS)
    return None


def stage_detect(args, cfg):
    clips = my_clips(args)

    for clip in clips:
        name = Path(clip).stem
        out = Path(args.work) / "clips" / name
        flag = out / "done.json"
        if flag.exists() and not args.force:
            log(f"  {name}: already done, skipping (use --force to redo)")
            continue

        out.mkdir(parents=True, exist_ok=True)
        # A re-run renumbers the tracks, so an old reject list would reject the
        # wrong ones. Move it aside rather than delete it: it is your work.
        old_rej = out / "reject.txt"
        if old_rej.exists():
            moved = out / f"reject.before-{time.strftime('%Y%m%d-%H%M%S')}.txt"
            old_rej.rename(moved)
            log(f"  {name}: moved the old reject.txt to {moved.name} -- track IDs "
                f"change on a re-run; review the new sheets and write a new one")

        # STREAMED, never materialised. At 3840x2160 a frame is 25 MB, so a
        # 300-frame list is 7.5 GB -- and the audit used to keep a second copy.
        # Detection only ever needs an 11-frame rolling window (275 MB), and
        # every other consumer here can take a second pass over the file.
        src = F.VideoSource(clip)

        def stream(limit=args.max_frames):
            for i, f in enumerate(src):
                if limit and i >= limit:
                    return
                yield f

        first = next(iter(stream(1)), None)
        if first is None:
            log(f"  {name}: no frames decoded, skipping")
            continue
        H, W = first.shape[:2]
        n_frames = min(len(src) or 10 ** 9, args.max_frames or 10 ** 9)

        c = clip_config(cfg, W, H, src.fps)

        with Timer(f"  {name}: auditing"):
            audit = F.audit(_Gen(stream), name)
        json.dump(audit, open(out / "audit.json", "w"), indent=2, default=str)
        log(f"  {name}: {n_frames} frames {W}x{H} @ {c.track.fps:.1f}fps, "
            f"blue sep {audit['separability'].get('blue', 0):.1f}σ, "
            f"dup {audit['duplicate_fraction']:.0%}")
        for w in audit["warnings"]:
            log(f"    !! {w}")
        if audit["ir_night_mode"]:
            c.detect.channel = "gray"

        roi = roi_for(args, first)
        if roi is not None:
            cv2.imwrite(str(out / "roi_mask.png"), F.overlay_mask(first, roi))

        with Timer(f"  {name}: detecting"):
            dets = D.detect_source(_Gen(stream), c.detect, roi,
                                   progress=progress_printer(n_frames))

        ratio, diag = T.estimate_step_ratio(dets, c.track, return_diagnostics=True)
        c.track.step_ratio = ratio
        log(f"  {name}: step_ratio {ratio:.2f} (sharpness {diag['sharpness']:.1f})")
        tracks = T.link(dets, c.track, step_ratio=ratio)
        acc = T.accept(tracks, c.track)
        rej = T.rejected(tracks, acc, c.track)
        log(f"  {name}: {sum(len(v) for v in dets.values())} candidates -> "
            f"{len(acc)} accepted tracks, {len(rej)} rejected")

        pd.DataFrame(T.summarize(acc, c.track)).to_csv(out / "tracks.csv", index=False)
        pd.DataFrame([{**d.as_dict(), "track_id": t.id} for t in acc for d in t.dets]) \
            .to_csv(out / "detections.csv", index=False)
        if rej:
            pd.DataFrame(T.summarize(rej, c.track)).to_csv(out / "negatives.csv",
                                                           index=False)
        with open(out / "tracks.pkl", "wb") as fh:
            pickle.dump({"accepted": acc, "rejected": rej}, fh)

        # Second pass: contact-sheet crops, the heatmap frame, background
        # frames and the overlay video, all from one stream.
        with Timer(f"  {name}: review artefacts"):
            R.make_contact_sheets_streaming(acc, _Gen(stream), out / "review")

            bgdir = out / "bg"
            bgdir.mkdir(exist_ok=True)
            by = {}
            for tr in acc:
                for d in tr.dets:
                    by.setdefault(d.frame, []).append(d)
            vw = None
            if args.overlay:
                vw = cv2.VideoWriter(str(out / "overlay.mp4"),
                                     cv2.VideoWriter_fourcc(*"mp4v"),
                                     c.track.fps, (W, H))
            mid = n_frames // 2
            for i, f in enumerate(stream()):
                if i < 12:
                    cv2.imwrite(str(bgdir / f"{i:03d}.png"), f)
                if i == mid:
                    cv2.imwrite(str(out / "fp_heatmap.png"),
                                R.fp_heatmap(rej, acc, f))
                if vw is not None:
                    vw.write(D.draw(f, by.get(i, [])))
            if vw is not None:
                vw.release()

        c.save(out / "config.json")
        json.dump({"clip": clip, "n_frames": n_frames, "shape": [H, W],
                   "fps": c.track.fps, "n_accepted": len(acc),
                   "n_rejected": len(rej), "step_ratio": ratio},
                  open(flag, "w"), indent=2)

    log("detect stage complete", rule=True)
    log("Review the contact sheets in <work>/clips/*/review/, then put the track")
    log("IDs that are NOT firebrands in <work>/clips/<clip>/reject.txt")
    log("(on Batch: bash scripts/05-gcs_sync.sh review, edit, then")
    log(" bash scripts/03-upload_inputs.sh to push the reject.txt files up)")


# ---------------------------------------------------------------------------
# stage: tiles
# ---------------------------------------------------------------------------

def read_rejects(d: Path):
    f = d / "reject.txt"
    if not f.exists():
        return set()
    txt = f.read_text().replace(",", " ").split()
    return {int(x) for x in txt if x.strip().lstrip("-").isdigit()}


# ---------------------------------------------------------------------------
# real footage with NO firebrands: <work>/negatives.txt
# ---------------------------------------------------------------------------

def read_negatives(path: Path):
    """<work>/negatives.txt -> {clip_stem: [(first, last), ...]}, None = whole clip.

        # clip   frames (0-based, inclusive) where NO firebrand is anywhere in view
        clip3    0-150
        clip5    400-520, 900-1010
        clip8                      # the whole clip

    These are the only frames the kit can label with certainty without a
    person drawing anything: if nothing is there, every pixel is "not a
    firebrand". suggest_negatives.py proposes ranges; you confirm by watching.
    """
    out = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.replace(",", " ").split()
        name, ranges = Path(parts[0]).stem, []
        for tok in parts[1:]:
            a, sep, b = tok.partition("-")
            if not a.isdigit() or (sep and not b.isdigit()):
                raise SystemExit(f"negatives.txt: cannot read {tok!r} in {line!r} -- "
                                 f"use FIRST-LAST frame numbers, e.g. 120-300")
            lo, hi = int(a), int(b) if sep else int(a)
            ranges.append((min(lo, hi), max(lo, hi)))
        out.setdefault(name, []).extend(ranges or [None])
    return out


def iter_negative_frames(args, cfg, specs, max_frames, work=None):
    """Yield (clip, segment, frame_index, residual_stack, roi, clip_cfg) for the
    frames listed in negatives.txt, at most `max_frames` of them in total.

    Streamed one frame at a time: a 4K stack is 25 MB, so holding a few hundred
    would not fit. Decoding starts half a median window early so the first
    listed frame already has a real background behind it.
    """
    clips = {Path(c).stem: c for c in list_clips(args.clips)}
    budget, seg = max_frames, 0
    for name in sorted(specs):
        if budget <= 0:
            return
        if name not in clips:
            log(f"  negatives.txt: no clip called {name} in {args.clips} -- skipped")
            continue
        src = F.VideoSource(clips[name])
        first = next(iter(src), None)
        if first is None:
            continue
        H, W = first.shape[:2]
        c = clip_config(cfg, W, H, src.fps)
        det_cfg = Path(work or args.work) / "clips" / name / "config.json"
        if det_cfg.exists():
            c.detect.channel = Config.load(det_cfg).detect.channel
        elif F.is_ir(first):
            c.detect.channel = "gray"
        roi = roi_for(args, first)
        n_total = len(src) or 10 ** 9
        for r in sorted(specs[name], key=lambda r: (-1, 0) if r is None else r):
            if budget <= 0:
                return
            a, b = (0, n_total - 1) if r is None else r
            b = min(b, a + budget - 1)
            lo = max(0, a - c.detect.median_window)

            def window(lo=lo, hi=b):
                for i, f in enumerate(src):
                    if i > hi:
                        return
                    if i >= lo:
                        yield f

            for k, stack in DS.residual_stacks(window(), c.detect, roi,
                                               cfg.train.n_frames_stack):
                fi = lo + k
                if a <= fi <= b and budget > 0:
                    budget -= 1
                    yield name, seg, fi, stack, roi, c
            seg += 1


def count_negative_frames(args, specs):
    clips = {Path(c).stem: c for c in list_clips(args.clips)}
    n = 0
    for name, ranges in specs.items():
        if name not in clips:
            continue
        total = None
        for r in ranges:
            if r is None:
                total = total or len(F.VideoSource(clips[name])) or 0
                n += total
            else:
                n += r[1] - r[0] + 1
    return n


def read_run(args, clip_stem, first, n):
    """n consecutive raw frames of one clip, from `first` on."""
    path = {Path(c).stem: c for c in list_clips(args.clips)}.get(clip_stem)
    if path is None:
        return []
    out = []
    for i, f in enumerate(F.VideoSource(path)):
        if i >= first + n:
            break
        if i >= first:
            out.append(f)
    return out


def stage_tiles(args, cfg):
    work = Path(args.work)
    tiles_path = work / "data" / "tiles.npz"
    if tiles_path.exists() and not args.force:
        log(f"tiles already built ({tiles_path}), skipping")
        return

    clip_dirs = sorted(d for d in (work / "clips").iterdir() if d.is_dir()) \
        if (work / "clips").exists() else []
    if not clip_dirs:
        raise SystemExit("no detect output found -- run --stage detect first")

    verified_rows, bg_runs, ratios, footage_peaks = [], [], [], []
    n_rejected_manual = 0
    for d in clip_dirs:
        # done.json is written last, so a clip whose detect run was cut off
        # (tracks.pkl but no done.json) is skipped rather than half-used.
        if not (d / "done.json").exists():
            if (d / "tracks.pkl").exists():
                log(f"  {d.name}: detect did not finish, skipping -- re-run detect")
            continue
        with open(d / "tracks.pkl", "rb") as fh:
            blob = pickle.load(fh)
        rej_ids = read_rejects(d)
        n_rejected_manual += len(rej_ids)
        keep = [t for t in blob["accepted"] if t.id not in rej_ids]
        # brightness of every coherent track in the footage, kept or not --
        # used only to set the generator's upper brightness bound (see synth)
        footage_peaks += [x.peak for t in blob["accepted"] + blob["rejected"]
                          if len(t.dets) >= cfg.track.min_track_len for x in t.dets]
        meta = json.load(open(d / "done.json"))
        c = Config.load(d / "config.json")
        verified_rows += T.summarize(keep, c.track)
        ratios.append(meta.get("step_ratio"))
        bg_paths = sorted((d / "bg").glob("*.png")) if (d / "bg").exists() else []
        if bg_paths:
            bg_runs.append((d.name, bg_paths))
        log(f"  {d.name}: {len(keep)} verified "
            f"({len(rej_ids)} manually rejected)")

    if not verified_rows:
        raise SystemExit("no verified tracks -- check the detect stage output")
    if not n_rejected_manual:
        log("  !! no reject.txt anywhere. That is fine to start, but the")
        log("     generator is being fitted to unreviewed tracks.")

    cfg.synth = S.fit_config_to_tracks(
        verified_rows, cfg.synth,
        step_ratio=float(np.median([r for r in ratios if r])) if ratios else None,
        extra_peaks=footage_peaks)

    if not bg_runs:
        raise SystemExit("no background frames saved -- re-run detect")
    first_bg = cv2.imread(str(bg_runs[0][1][0]))
    H, W = first_bg.shape[:2]
    cfg.detect.tophat_ksize = max(5, int(round(15 * W / 1920)) | 1)
    # Streak width scales with resolution too. Keep the thin end (faint, far
    # embers) and widen the other end, from the defaults so reruns do not
    # compound. A --set synth.psf_sigma=... override wins.
    if "synth.psf_sigma" not in args.overridden:
        lo, hi = SynthConfig().psf_sigma
        cfg.synth.psf_sigma = (lo, hi * px_scale(W))
    log(f"  fitted speed_px {tuple(round(v,1) for v in cfg.synth.speed_px)}, "
        f"peak_srgb {tuple(round(v,1) for v in cfg.synth.peak_srgb)}, "
        f"psf_sigma {tuple(round(v,2) for v in cfg.synth.psf_sigma)}, "
        f"exposure_duty {cfg.synth.exposure_duty:.3f}")
    roi = F.mask_from_spec((H, W), F.load_mask_spec(args.mask)) if args.mask else None

    # Backgrounds: one run per clip (every clip, not just the first few), and
    # -- better -- runs from frames you listed as having NO firebrands, which
    # carry the camera's real noise without real embers that would otherwise
    # be in the picture but labelled "not a firebrand". Half the chunks use
    # those clean runs when there are any.
    def loader(paths):
        return lambda: [cv2.imread(str(q)) for q in paths]

    runs = [loader(pths) for _name, pths in bg_runs
            if cv2.imread(str(pths[0])).shape[:2] == (H, W)]
    specs = read_negatives(work / "negatives.txt")
    clean = []
    for name in sorted(specs):
        for r in specs[name][:2]:
            if len(clean) >= 4:
                break
            a = 0 if r is None else r[0]
            n = 16 if r is None else min(16, r[1] - r[0] + 1)
            frames = read_run(args, name, a, n) if n >= 3 else []
            if frames and frames[0].shape[:2] == (H, W):
                clean.append(frames)
    if clean:
        mixed = []
        for i in range(max(len(runs), len(clean))):
            mixed += [clean[i % len(clean)], runs[i % len(runs)]]
        runs = mixed
    log(f"  synthetic backgrounds: {len(bg_runs)} clip run(s)"
        + (f" + {len(clean)} firebrand-free run(s) from negatives.txt" if clean else
           "  (no negatives.txt -- backgrounds contain real, unlabelled embers)"))

    with Timer(f"  rendering {args.synth_frames} synthetic frames"):
        X, Y, _sf, _sg = DS.build_from_synthetic(runs, args.synth_frames, cfg,
                                                 seed=0, roi_mask=roi)
    log(f"  synthetic: {DS.tile_stats(X, Y)}")

    # Real negatives: tiles of footage you confirmed has no firebrands, target
    # all zero. Without them the model has only ever seen scenes WITH embers
    # and fires on texture and noise when a clip goes quiet.
    if specs:
        n_target = int(round(len(X) * args.neg_frac / max(1e-6, 1 - args.neg_frac)))
        rng = np.random.default_rng(1)
        Xn, used_frames, tile = [], 0, cfg.train.tile
        avail = min(count_negative_frames(args, specs), args.max_neg_frames)
        per_frame = max(1, min(48, -(-n_target // max(1, avail))))
        with Timer(f"  real negatives: up to {n_target} tiles from negatives.txt"):
            for _clip, _seg, _fi, stack, nroi, _c in iter_negative_frames(
                    args, cfg, specs, args.max_neg_frames, work):
                if len(Xn) >= n_target:
                    break
                used_frames += 1
                pos = DS.tile_positions(stack.shape[1:], tile, cfg.train.stride)
                rng.shuffle(pos)
                took = 0
                for (y, x) in pos:
                    if took >= per_frame or len(Xn) >= n_target:
                        break
                    if nroi is not None and (nroi[y:y + tile, x:x + tile] > 0).mean() < 0.5:
                        continue
                    Xn.append(stack[:, y:y + tile, x:x + tile])
                    took += 1
        if Xn:
            Xn = np.array(Xn, np.uint8)
            X = np.concatenate([X, Xn])
            Y = np.concatenate([Y, np.zeros((len(Xn),) + Y.shape[1:], Y.dtype)])
            log(f"  real negatives: {len(Xn)} tiles from {used_frames} firebrand-free "
                f"frames ({len(Xn) / len(X):.0%} of all tiles)")
        else:
            log("  !! negatives.txt is there but gave no tiles -- check clip names and frames")
    else:
        log("  !! no negatives.txt: the model will never see a scene with no firebrands.")
        log("     List some firebrand-free frames (python suggest_negatives.py helps).")
    log(f"  all tiles: {DS.tile_stats(X, Y)}")
    tiles_path.parent.mkdir(parents=True, exist_ok=True)
    DS.save_tiles(tiles_path, X, Y)
    cfg.save(work / "config.json")
    log(f"tiles written to {tiles_path}", rule=True)


# ---------------------------------------------------------------------------
# stage: train
# ---------------------------------------------------------------------------

def stage_train(args, cfg):
    work = Path(args.work)
    run_dir = work / "runs" / "streak"
    ck = run_dir / "best.pt"
    # done.json, NOT best.pt, marks a finished run: best.pt appears after the
    # first epoch, so an interrupted run would otherwise be skipped forever.
    if (run_dir / "done.json").exists() and not args.force:
        log(f"training already finished ({run_dir / 'done.json'}), skipping")
        return
    tiles_path = work / "data" / "tiles.npz"
    if not tiles_path.exists():
        raise SystemExit("no tiles -- run --stage tiles first")

    from firebrand import train as TR
    if args.force:
        for f in ("last.pt", "done.json"):
            (run_dir / f).unlink(missing_ok=True)
    # as_float=False keeps masks uint8; see load_tiles. With float masks the
    # array pair is ~3.4 GB instead of ~1.9 GB at 7k tiles.
    X, Y = DS.load_tiles(tiles_path, as_float=False)
    log(f"  {len(X)} tiles, {X.shape[1:]} each, "
        f"{(X.nbytes + Y.nbytes)/1e9:.2f} GB resident")
    cfg.train.epochs = args.epochs
    with Timer(f"  training {args.epochs} epochs"):
        TR.train(X, Y, cfg.train, out_dir=run_dir, log_every=1,
                 resume=not args.force)
    log(f"checkpoint at {ck}", rule=True)


# ---------------------------------------------------------------------------
# stage: eval
# ---------------------------------------------------------------------------

def stage_eval(args, cfg):
    work = Path(args.work)
    ck = work / "runs" / "streak" / "best.pt"
    if not ck.exists():
        raise SystemExit("no checkpoint -- run --stage train first")
    report_path = work / "runs" / "streak" / "report.json"
    if report_path.exists() and not args.force:
        log(f"report already exists ({report_path}), skipping")
        return

    import torch
    from firebrand import train as TR, infer as I

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, tcfg = TR.load(ck, device=dev)
    cfg.train = tcfg                 # tile/stride/stack the model was trained with

    clip_dirs = sorted(d for d in (work / "clips").iterdir() if d.is_dir())
    bgd = next((d / "bg" for d in clip_dirs if (d / "bg").exists()), None)
    if bgd is None:
        raise SystemExit("no background frames for the held-out set")
    bg = [cv2.imread(str(q)) for q in sorted(bgd.glob("*.png"))]

    H, W = bg[0].shape[:2]
    cfg.detect.tophat_ksize = max(5, int(round(15 * W / 1920)) | 1)
    roi = F.mask_from_spec((H, W), F.load_mask_spec(args.mask)) if args.mask else None

    # Seed 999: never used to build the training tiles, so this is held out.
    with Timer("  rendering held-out evaluation clip"):
        tf, tg = S.make_clip(bg, 60, cfg.synth, seed=999, roi_mask=roi)
    gtf = E.filter_gt(tg, min_L=cfg.detect.min_length, min_peak_srgb=25.0)

    with Timer("  threshold sweep"):
        sweep = I.sweep_threshold(model, F.ArraySource(tf), cfg, gtf, roi, device=dev)
    best_f1 = max(sweep, key=lambda r: r["f1"])
    best = best_f1
    log(f"  best-F1 threshold on synthetic embers: {best_f1['threshold']} "
        f"(F1 {best_f1['f1']:.3f})")

    # The synthetic sweep always has embers in view, so it cannot see the
    # failure that matters on quiet footage: tagging texture when nothing is
    # there. If negatives.txt lists firebrand-free frames, count the tracks the
    # model invents on them at every threshold, and pick the best-F1 threshold
    # that keeps that under --max-empty-fp per 100 frames.
    specs = read_negatives(work / "negatives.txt")
    empty = None
    if specs:
        ths = [r["threshold"] for r in sweep]
        dets = {th: {} for th in ths}
        seg_cfg = {}
        n_used = 0
        with Timer("  false tracks on firebrand-free frames"):
            for clip, seg, fi, stack, nroi, c in iter_negative_frames(
                    args, cfg, specs, args.max_eval_neg_frames, work):
                c.train = cfg.train
                seg_cfg[(clip, seg)] = c
                prob = I.predict_map(model, stack, cfg.train.tile, cfg.train.stride, dev)
                for th in ths:
                    dets[th].setdefault((clip, seg), {})[fi] = \
                        I.detections_from_map(prob, fi, c, th, nroi)
                n_used += 1
        rates = {}
        for th in ths:
            n_tr = 0
            for key, by in dets[th].items():
                c = seg_cfg[key]
                n_tr += len(T.accept(T.link(by, c.track,
                                            step_ratio=T.estimate_step_ratio(by, c.track)),
                                     c.track))
            rates[th] = 100.0 * n_tr / max(n_used, 1)
        for r in sweep:
            r["false_tracks_per_100_empty_frames"] = rates[r["threshold"]]
        empty = {"frames": n_used, "max_allowed": args.max_empty_fp}
        log(f"  {n_used} firebrand-free frames: false tracks per 100 frames")
        for r in sweep:
            log(f"    threshold {r['threshold']:.1f}   F1 {r['f1']:.3f}   "
                f"false tracks {rates[r['threshold']]:.2f}")
        ok = [r for r in sweep if rates[r["threshold"]] <= args.max_empty_fp]
        best = max(ok, key=lambda r: r["f1"]) if ok else \
            min(sweep, key=lambda r: (rates[r["threshold"]], -r["f1"]))
        if best is not best_f1:
            log(f"  -> using threshold {best['threshold']} instead: the best-F1 one "
                f"invents {rates[best_f1['threshold']]:.2f} tracks per 100 empty frames")
    log(f"  operating threshold {best['threshold']} (F1 {best['f1']:.3f})")

    with Timer("  learned model"):
        m_dets, m_tracks = I.run_clip(model, F.ArraySource(tf), cfg, roi,
                                      best["threshold"], dev)
    with Timer("  classical detector"):
        c_dets = D.detect_source(F.ArraySource(tf), cfg.detect, roi)
        c_acc = T.accept(T.link(c_dets, cfg.track,
                                step_ratio=T.estimate_step_ratio(c_dets, cfg.track)),
                         cfg.track)

    rep = {"threshold": best["threshold"], "threshold_best_f1": best_f1["threshold"],
           "empty_frames": empty, "sweep": sweep,
           "learned": E.report(m_dets, m_tracks, gtf),
           "classical": E.report(c_dets, c_acc, gtf)}
    with open(report_path, "w") as fh:
        json.dump(rep, fh, indent=2, default=float)

    log("", rule=True)
    lt = rep["learned"]["trk_track_recall"]
    ct = rep["classical"]["trk_track_recall"]
    log(f"  track recall   classical {ct:.3f}   learned {lt:.3f}")
    log(f"  detection P/R  classical {rep['classical']['det_precision']:.3f}/"
        f"{rep['classical']['det_recall']:.3f}   learned "
        f"{rep['learned']['det_precision']:.3f}/{rep['learned']['det_recall']:.3f}")
    if lt > ct:
        log(f"  The model beats the classical detector by {lt - ct:+.3f} track recall.")
    else:
        log("  The model does NOT beat the classical detector yet. The fix is")
        log("  almost always more or better-matched synthetic data (raise")
        log("  --synth-frames, and verify more tracks), not a bigger model.")
    log(f"  full report: {report_path}")


# ---------------------------------------------------------------------------
# stage: tag -- the point of the whole exercise
# ---------------------------------------------------------------------------

def stage_tag(args, cfg):
    """Run the trained model over every real clip and write the firebrands it
    finds, in the same formats the classical detector writes.

    Output per clip in <work>/tags/<clip>/:
        tracks.csv       one row per accepted track (ballistic residual included)
        detections.csv   one row per detection, with track_id
        labels/*.txt     YOLO boxes, one file per frame that has any
        overlay.mp4      (with --overlay) detections drawn on the clip
        done.json        counts, threshold, and the classical count to compare

    The acceptance rule still applies to the model's output: even a good model
    makes isolated false positives, and 3-frame coherence is the cheapest way
    to remove them.
    """
    work = Path(args.work)
    ck = work / "runs" / "streak" / "best.pt"
    if not ck.exists():
        raise SystemExit("no trained model -- run the train step first")

    thresh = args.threshold
    rep_path = work / "runs" / "streak" / "report.json"
    if thresh is None and rep_path.exists():
        thresh = float(json.load(open(rep_path))["threshold"])
        log(f"  threshold {thresh} (picked by the eval stage)")
    elif thresh is None:
        thresh = 0.5
        log("  threshold 0.5 (no report.json -- run eval to pick one properly)")

    import torch
    from firebrand import train as TR, infer as I
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, tcfg = TR.load(ck, device=dev)
    log(f"  model {ck} on {dev}")
    if dev == "cpu":
        log("  !! no GPU -- tagging at 4K on CPU is very slow")

    for clip in my_clips(args):
        name = Path(clip).stem
        out = work / "tags" / name
        flag = out / "done.json"
        if flag.exists() and not args.force:
            log(f"  {name}: already tagged, skipping (use --force to redo)")
            continue
        out.mkdir(parents=True, exist_ok=True)

        src = F.VideoSource(clip)

        def stream(limit=args.max_frames):
            for i, f in enumerate(src):
                if limit and i >= limit:
                    return
                yield f

        first = next(iter(stream(1)), None)
        if first is None:
            log(f"  {name}: no frames decoded, skipping")
            continue
        H, W = first.shape[:2]
        n_frames = min(len(src) or 10 ** 9, args.max_frames or 10 ** 9)

        # Current settings (defaults + --set), plus what detect MEASURED on this
        # clip: fps, the IR fallback and the calibrated step ratio. The
        # thresholds themselves come from now, not from the detect run, so a
        # retuned tracker applies without re-running detect.
        c = clip_config(cfg, W, H, src.fps)
        det_cfg = work / "clips" / name / "config.json"
        if det_cfg.exists():
            m = Config.load(det_cfg)
            c.track.fps, c.track.step_ratio = m.track.fps, m.track.step_ratio
            c.detect.channel = m.detect.channel
        elif F.is_ir(first):
            c.detect.channel = "gray"
        c.train = tcfg
        roi = roi_for(args, first)

        with Timer(f"  {name}: tagging {n_frames} frames at threshold {thresh}"):
            dets, acc = I.run_clip(model, _Gen(stream), c, roi, thresh, dev,
                                   progress=progress_printer(n_frames))
        if args.min_track_conf > 0:
            # Hysteresis: detect at the operating threshold (keeps the faint
            # parts of real tracks), but keep a track only if the model was
            # confident on at least one of its detections. Texture the model
            # half-believes never gets there. Detection.peak here is the
            # model's probability x 255.
            before = len(acc)
            acc = [t for t in acc
                   if max(d.peak for d in t.dets) >= 255.0 * args.min_track_conf]
            log(f"  {name}: --min-track-conf {args.min_track_conf}: kept "
                f"{len(acc)} of {before} tracks")

        pd.DataFrame(T.summarize(acc, c.track)).to_csv(out / "tracks.csv", index=False)
        pd.DataFrame([{**d.as_dict(), "track_id": t.id} for t in acc for d in t.dets]) \
            .to_csv(out / "detections.csv", index=False)
        # Everything the model detected that did NOT end up in an accepted
        # track. When the overlay misses a firebrand, this says why: a yellow
        # box on it means the model saw it and the tracker dropped it; no box
        # means the model never saw it (threshold, or training data).
        in_acc = {id(d) for t in acc for d in t.dets}
        cand = [d for v in dets.values() for d in v if id(d) not in in_acc]
        pd.DataFrame([d.as_dict() for d in cand], columns=list(D.Detection.__dataclass_fields__)) \
            .to_csv(out / "candidates.csv", index=False)
        lab = out / "labels"
        lab.mkdir(exist_ok=True)
        for fi, boxes in T.to_yolo_labels(acc, (H, W)).items():
            with open(lab / f"{fi:06d}.txt", "w") as fh:
                for cl, xc, yc, bw, bh in boxes:
                    fh.write(f"{cl} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

        if args.overlay:
            by, cby = {}, {}
            for tr in acc:
                for d in tr.dets:
                    by.setdefault(d.frame, []).append(d)
            for d in cand:
                cby.setdefault(d.frame, []).append(d)
            vw = cv2.VideoWriter(str(out / "overlay.mp4"),
                                 cv2.VideoWriter_fourcc(*"mp4v"), c.track.fps, (W, H))
            for i, f in enumerate(stream()):
                f = D.draw(f, cby.get(i, []), color=(0, 220, 255), thickness=1)   # yellow
                vw.write(D.draw(f, by.get(i, [])))                                # green
            vw.release()

        classical = None
        det_done = work / "clips" / name / "done.json"
        if det_done.exists():
            classical = json.load(open(det_done)).get("n_accepted")
        n_det = sum(len(v) for v in dets.values())
        log(f"  {name}: {n_det} model detections -> {len(acc)} accepted tracks, "
            f"{len(cand)} detections not in any accepted track"
            + (f" (classical detector: {classical})" if classical is not None else ""))
        with open(flag, "w") as fh:
            json.dump({"clip": clip, "n_frames": n_frames, "shape": [H, W],
                       "threshold": thresh, "n_detections": n_det,
                       "n_tracks": len(acc), "classical_n_accepted": classical,
                       "model": str(ck)}, fh, indent=2)

    log("tag stage complete", rule=True)


# ---------------------------------------------------------------------------

STAGES = {"detect": stage_detect, "tiles": stage_tiles,
          "train": stage_train, "eval": stage_eval, "tag": stage_tag}
ALL = ["detect", "tiles", "train", "eval", "tag"]
PER_CLIP = {"detect", "tag"}


def apply_overrides(cfg, items):
    """--set track.max_heading_change_deg=40 -> cfg.track.max_heading_change_deg = 40.0"""
    done = set()
    for item in items:
        key, eq, raw = item.partition("=")
        section, dot, field = key.strip().partition(".")
        sub = getattr(cfg, section, None) if dot else None
        if not eq or sub is None or not hasattr(sub, field):
            raise SystemExit(f"bad --set {item!r}: want SECTION.KEY=VALUE with SECTION one of "
                             f"detect, track, synth, train and KEY a field in firebrand/config.py")
        try:
            val = json.loads(raw)
        except ValueError:
            val = raw
        cur = getattr(sub, field)
        if isinstance(cur, bool):
            val = str(val).lower() in ("1", "true", "yes")
        elif isinstance(cur, float) and isinstance(val, (int, float)):
            val = float(val)
        elif isinstance(cur, tuple) and isinstance(val, list):
            val = tuple(val)
        setattr(sub, field, val)
        done.add(f"{section}.{field}")
        log(f"  setting {section}.{field} = {val!r}")
    return done


def parse_stages(spec):
    if spec == "all":
        return list(ALL)
    names = [s.strip() for s in spec.split(",") if s.strip()]
    bad = [n for n in names if n not in STAGES]
    if bad or not names:
        raise SystemExit(f"unknown stage(s) {bad or spec!r}; choose from "
                         f"{', '.join(ALL)} or all")
    return sorted(names, key=ALL.index)          # always run in pipeline order


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", default="clips/",
                    help="folder of videos, one video, or a gs:// prefix")
    ap.add_argument("--work", default="work",
                    help="output directory; on Batch this is inside the bucket")
    ap.add_argument("--mask", help="mask JSON from make_mask.py")
    ap.add_argument("--no-mask", action="store_true")
    ap.add_argument("--stage", default="all",
                    help=f"one of {', '.join(ALL)}, a comma list, or all")
    ap.add_argument("--shard", default=None,
                    help="I/N to process every Nth clip starting at I, or 'auto' "
                         "to read BATCH_TASK_INDEX/BATCH_TASK_COUNT")
    ap.add_argument("--force", action="store_true",
                    help="redo stages whose output already exists")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--synth-frames", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--threshold", type=float, default=None,
                    help="tag: model probability threshold (default: from eval)")
    ap.add_argument("--overlay", action="store_true")
    ap.add_argument("--neg-frac", type=float, default=0.25,
                    help="tiles: share of training tiles taken from firebrand-free "
                         "frames listed in <work>/negatives.txt")
    ap.add_argument("--max-neg-frames", type=int, default=600,
                    help="tiles: most firebrand-free frames to read")
    ap.add_argument("--max-eval-neg-frames", type=int, default=200,
                    help="eval: firebrand-free frames used to count false tracks")
    ap.add_argument("--max-empty-fp", type=float, default=1.0,
                    help="eval: most false tracks per 100 firebrand-free frames "
                         "allowed when choosing the operating threshold")
    ap.add_argument("--min-track-conf", type=float, default=0.0,
                    help="tag: keep a track only if the model's probability "
                         "reaches this on at least one of its detections (0 = off)")
    ap.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE",
                    help="override any setting in firebrand/config.py, e.g. "
                         "--set track.max_heading_change_deg=40 (repeatable)")
    args = ap.parse_args(argv)
    args.shard = parse_shard(args.shard)

    Path(args.work).mkdir(parents=True, exist_ok=True)
    # Detection and tracking settings come from the code (plus --set), NOT from
    # <work>/config.json: that file is written by the tiles stage and used to
    # carry the whole config, so an old run's thresholds silently overrode any
    # later fix to the defaults. Only the fitted generator is carried over.
    cfg = Config()
    cfg_path = Path(args.work) / "config.json"
    if cfg_path.exists():
        cfg.synth = Config.load(cfg_path).synth
    args.overridden = apply_overrides(cfg, args.set)

    order = parse_stages(args.stage)
    log(f"stages: {' -> '.join(order)}   work={args.work}", rule=True)

    failed = None
    for st in order:
        log(f"STAGE: {st}", rule=True)
        if st not in PER_CLIP and args.shard[0] != 0:
            log(f"  {st} needs every clip at once; shard 0 runs it, this is "
                f"shard {args.shard[0]} -- skipping")
            continue
        try:
            STAGES[st](args, cfg)
        except SystemExit as e:
            log(f"STAGE {st} stopped: {e}")
            failed = st
            break
        except Exception:
            log(f"STAGE {st} FAILED:")
            traceback.print_exc()
            failed = st
            break

    log("", rule=True)
    if failed:
        log(f"Stopped at '{failed}'. Everything before it is saved in {args.work} --")
        log("fix the problem and re-run; completed stages are skipped.")
        return 1
    log("All requested stages complete.")
    w = args.work
    for st, what in (("detect", f"{w}/clips/*/tracks.csv, review/ sheets"),
                     ("tiles", f"{w}/data/tiles.npz"),
                     ("train", f"{w}/runs/streak/best.pt"),
                     ("eval", f"{w}/runs/streak/report.json"),
                     ("tag", f"{w}/tags/*/tracks.csv")):
        if st in order:
            log(f"  {st:7s} {what}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        log("interrupted -- completed stages are saved and will be skipped on re-run")
        raise SystemExit(130)
