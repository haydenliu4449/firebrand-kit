#!/usr/bin/env python3
"""How good are the model's tags on your REAL footage?

    bash scripts/05-gcs_sync.sh review      # classical tracks (+ your reject.txt stays local)
    bash scripts/05-gcs_sync.sh tags        # the model's tags
    python evaluate_tags.py                 # every tagged clip
    python evaluate_tags.py --clip clip5 --overlay
    python evaluate_tags.py --heldout       # only frames training never used

The eval step (runs/streak/report.json) scores the model on SYNTHETIC embers,
where ground truth is exact. This scores it on your real clips, where the only
ground truth is what you verified by eye -- so it is built around the review
you already did, plus one more round of it.

HOW IT DECIDES

Your verified tracks are the classical detector's accepted tracks minus the
IDs in work/clips/<clip>/reject.txt. Precision 1.00 on the benchmark is what
makes them usable as ground truth, and your review is what makes them trusted.
A model track and a classical track are the same ember when at least half of
one's detections have a partner in the other within 5 px, frame by frame (the
kit's standard: centre distance, never IoU, at 7 px).

Every model track then lands in one of three bins:

  confirmed     matches a verified track                       -> true positive
  known false   matches a track YOU rejected in the classical  -> false positive
                review (the model learned a confuser)
  model-only    matches nothing the classical detector kept    -> unknown

Model-only is the interesting bin: it is either an ember the classical
detector missed (the reason the model exists -- faint embers on the ground)
or a new kind of false positive. Only you can tell which, so the script draws
them as contact sheets in work/eval/<clip>/model_only/. Put the IDs that are
NOT firebrands in work/tags/<clip>/reject.txt and run it again: precision stops
being a range and becomes a number, and the rest of the bin is counted as new
finds.

Verified tracks the model did not recover are drawn in work/eval/<clip>/missed/,
so you can see what kind of ember it loses.

OUTPUT
    work/eval/summary.csv                one row per clip, plus TOTAL
    work/eval/<clip>/model_only/*.png    review these, then write tags/<clip>/reject.txt
    work/eval/<clip>/missed/*.png        verified embers the model lost
    work/eval/<clip>/overlay.mp4         (--overlay) green = both found it,
                                         orange = model only, red = model missed it

Only frames both runs covered are compared: if detect ran with --max-frames 300
and tag ran on the whole clip, the comparison is over the first 300 frames.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from firebrand import frames as F, gcsio
from firebrand.detect import Detection
from firebrand.evaluate import match_frame
from firebrand.negatives import is_heldout
from firebrand.track import Track

DET_COLS = ["frame", "x", "y", "L", "W", "theta", "peak", "area", "flux"]
GREEN, ORANGE, RED = (90, 255, 90), (0, 165, 255), (60, 60, 255)


def read_ids(path: Path):
    if not path.exists():
        return None
    txt = path.read_text().replace(",", " ").split()
    return {int(t) for t in txt if t.strip().lstrip("-").isdigit()}


def read_dets(path: Path):
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=DET_COLS + ["track_id"])
    df = pd.read_csv(path)
    return df if len(df) else pd.DataFrame(columns=DET_COLS + ["track_id"])


def n_frames(done: Path):
    try:
        return int(json.load(open(done))["n_frames"])
    except (OSError, KeyError, ValueError):
        return None


def to_tracks(df, ids=None):
    out = []
    for tid, g in df.groupby("track_id"):
        if ids is not None and int(tid) not in ids:
            continue
        g = g.sort_values("frame")
        dets = [Detection(**{k: (int(r[k]) if k in ("frame", "area") else float(r[k]))
                             for k in DET_COLS}) for r in g.to_dict("records")]
        out.append(Track(int(tid), dets))
    return out


def match(model, classical, max_dist=5.0):
    """Frame-by-frame one-to-one matching. Returns, for every row of each
    table, the track id of its partner in the other (or -1)."""
    m_partner = np.full(len(model), -1)
    c_partner = np.full(len(classical), -1)
    m_by_f = model.groupby("frame").indices
    c_by_f = classical.groupby("frame").indices
    for f in set(m_by_f) & set(c_by_f):
        mi, ci = m_by_f[f], c_by_f[f]
        pairs, _ = match_frame(model.iloc[mi][["x", "y"]].to_numpy(),
                               classical.iloc[ci][["x", "y"]].to_numpy(), max_dist)
        for a, b in pairs:
            m_partner[mi[a]] = classical.iloc[ci[b]]["track_id"]
            c_partner[ci[b]] = model.iloc[mi[a]]["track_id"]
    return m_partner, c_partner


def majority(partners, min_frac):
    """Track-level partner: the most common partner track, if it covers at
    least min_frac of this track's detections."""
    p = [int(x) for x in partners if x >= 0]
    if not p:
        return None, 0.0
    tid, cnt = Counter(p).most_common(1)[0]
    frac = cnt / len(partners)
    return (tid if frac >= min_frac else None), frac


def find_clip(clips_arg, tag_done):
    name = Path(json.load(open(tag_done)).get("clip", "")).name
    if not clips_arg or not name:
        return None
    if gcsio.is_gcs(clips_arg):
        return clips_arg.rstrip("/") + "/" + name
    p = Path(clips_arg) / name
    if p.exists():
        return str(p)
    stem = Path(name).stem
    hits = [q for q in Path(clips_arg).glob(stem + ".*")
            if q.suffix.lower() in F.VIDEO_EXTS] if Path(clips_arg).is_dir() else []
    return str(hits[0]) if hits else None


def evaluate_clip(work: Path, name: str, args):
    tag_dir, det_dir, out = work / "tags" / name, work / "clips" / name, work / "eval" / name
    counts = [x for x in (n_frames(tag_dir / "done.json"), n_frames(det_dir / "done.json")) if x]
    n = min(counts) if counts else 10 ** 9

    model = read_dets(tag_dir / "detections.csv")
    classical = read_dets(det_dir / "detections.csv")
    model = model[model.frame < n].reset_index(drop=True)
    classical = classical[classical.frame < n].reset_index(drop=True)
    if args.heldout:
        # Only the held-out blocks: frames that --real-positives never trains on.
        model = model[[is_heldout(name, f) for f in model.frame]].reset_index(drop=True)
        classical = classical[[is_heldout(name, f) for f in classical.frame]] \
            .reset_index(drop=True)
        last = n if n < 10 ** 9 else int(max(model.frame.max() if len(model) else 0,
                                              classical.frame.max() if len(classical) else 0)) + 1
        n = sum(is_heldout(name, f) for f in range(last))

    c_rejected = read_ids(det_dir / "reject.txt")
    m_rejected = read_ids(tag_dir / "reject.txt") or set()
    reviewed_classical = c_rejected is not None
    c_rejected = c_rejected or set()

    m_partner, c_partner = match(model, classical, args.max_dist)
    model = model.assign(partner=m_partner)
    classical = classical.assign(partner=c_partner)

    # classical side: which verified tracks did the model recover?
    verified, recovered, missed, frag = [], [], [], []
    for tid, g in classical.groupby("track_id"):
        tid = int(tid)
        if tid in c_rejected:
            continue
        verified.append(tid)
        partners = [int(p) for p in g.partner if p >= 0]
        if len(partners) / len(g) >= args.min_overlap:
            recovered.append(tid)
            frag.append(len(set(partners)))
        else:
            missed.append(tid)

    # model side: sort every model track into a bin
    confirmed, known_false, model_only = [], [], []
    for tid, g in model.groupby("track_id"):
        tid = int(tid)
        partner, _ = majority(g.partner.to_numpy(), args.min_overlap)
        if partner is None:
            model_only.append(tid)
        elif partner in c_rejected:
            known_false.append(tid)
        else:
            confirmed.append(tid)

    n_model = len(confirmed) + len(known_false) + len(model_only)
    reviewed_fp = set(model_only) & m_rejected
    new_finds = [t for t in model_only if t not in m_rejected]
    if (tag_dir / "reject.txt").exists():
        tp = len(confirmed) + len(new_finds)
        prec_lo = prec_hi = tp / n_model if n_model else float("nan")
    else:
        prec_lo = len(confirmed) / n_model if n_model else float("nan")
        prec_hi = (n_model - len(known_false)) / n_model if n_model else float("nan")

    # physics check, free: ballistic residual of the model's tracks
    ball = pd.read_csv(tag_dir / "tracks.csv") if (tag_dir / "tracks.csv").exists() \
        and (tag_dir / "tracks.csv").stat().st_size else pd.DataFrame()
    if len(ball) and "frame_start" in ball:
        ball = ball[ball.frame_start < n]
    res = ball["ballistic_rms_px"].to_numpy() if len(ball) else np.array([])

    row = dict(
        clip=name, frames_compared=int(n or 0),
        classical_reviewed=reviewed_classical, model_reviewed=(tag_dir / "reject.txt").exists(),
        verified_tracks=len(verified), recovered=len(recovered), missed=len(missed),
        recall_vs_verified=len(recovered) / len(verified) if verified else float("nan"),
        fragmentation=float(np.mean(frag)) if frag else float("nan"),
        model_tracks=n_model, confirmed=len(confirmed), known_false=len(known_false),
        model_only=len(model_only), model_only_rejected=len(reviewed_fp),
        new_finds=len(new_finds) if (tag_dir / "reject.txt").exists() else None,
        precision_low=prec_lo, precision_high=prec_hi,
        ballistic_median_px=float(np.median(res)) if len(res) else float("nan"),
        ballistic_p90_px=float(np.percentile(res, 90)) if len(res) else float("nan"),
    )

    # pictures
    if args.no_sheets and not args.overlay:
        return row
    clip = find_clip(args.clips, tag_dir / "done.json")
    if clip is None:
        print(f"  {name}: clip not found under {args.clips} -- skipping sheets/overlay "
              f"(pass --clips gs://BUCKET/clips to read it from the bucket)")
        return row
    from firebrand import review as R
    src = F.VideoSource(clip)

    def window():
        for i, f in enumerate(src):
            if i >= n:
                return
            yield f

    class W:                      # re-iterable, like run_all._Gen
        def __iter__(self):
            return iter(window())

    if not args.no_sheets:
        todo = [t for t in model_only if t not in m_rejected] if args.pending_only else model_only
        if todo:
            R.make_contact_sheets_streaming(to_tracks(model, set(todo)), W(), out / "model_only")
        if missed:
            R.make_contact_sheets_streaming(to_tracks(classical, set(missed)), W(), out / "missed")

    if args.overlay:
        import cv2
        from firebrand import detect as D
        colour = {**{t: GREEN for t in confirmed}, **{t: ORANGE for t in model_only},
                  **{t: RED for t in known_false}}
        m_by = defaultdict(list)
        for t in to_tracks(model):
            for d in t.dets:
                m_by[d.frame].append((d, colour.get(t.id, GREEN)))
        miss_by = defaultdict(list)
        for t in to_tracks(classical, set(missed)):
            for d in t.dets:
                miss_by[d.frame].append(d)
        vw = None
        for i, f in enumerate(W()):
            if vw is None:
                h, w = f.shape[:2]
                out.mkdir(parents=True, exist_ok=True)
                vw = cv2.VideoWriter(str(out / "overlay.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                     float(src.fps or 15), (w, h))
            for d, c in m_by.get(i, []):
                f = D.draw(f, [d], color=c)
            f = D.draw(f, miss_by.get(i, []), color=RED, pad=10)
            vw.write(f)
        if vw is not None:
            vw.release()
    return row


def fmt(x, pct=False):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "  -  "
    return f"{x:5.2f}" if isinstance(x, float) else f"{x:5d}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default="work")
    ap.add_argument("--clips", default="clips",
                    help="folder with the videos, or gs://BUCKET/clips (for sheets/overlay)")
    ap.add_argument("--clip", action="append", help="only these clips (repeatable)")
    ap.add_argument("--overlay", action="store_true", help="write a colour-coded overlay.mp4")
    ap.add_argument("--no-sheets", action="store_true", help="numbers only, no contact sheets")
    ap.add_argument("--pending-only", action="store_true",
                    help="model_only sheets: skip IDs already in tags/<clip>/reject.txt")
    ap.add_argument("--max-dist", type=float, default=5.0, help="match radius, px")
    ap.add_argument("--min-overlap", type=float, default=0.5,
                    help="fraction of a track's detections that must match")
    ap.add_argument("--heldout", action="store_true",
                    help="score only the held-out frames (every 4th 32-frame block), which "
                         "training with --real-positives never uses -- the fair before/after")
    args = ap.parse_args(argv)

    work = Path(args.work)
    tags = work / "tags"
    names = sorted(d.name for d in tags.iterdir() if (d / "done.json").exists()) \
        if tags.exists() else []
    if args.clip:
        names = [n for n in names if n in set(args.clip)]
    if not names:
        raise SystemExit(f"no tagged clips in {tags}/ -- pull them first:\n"
                         f"  bash scripts/05-gcs_sync.sh tags\n"
                         f"  bash scripts/05-gcs_sync.sh review")

    rows = []
    for name in names:
        print(f"{name} ...", flush=True)
        rows.append(evaluate_clip(work, name, args))

    df = pd.DataFrame(rows)
    tot = {c: df[c].sum() for c in ("frames_compared", "verified_tracks", "recovered",
                                    "missed", "model_tracks", "confirmed", "known_false",
                                    "model_only", "model_only_rejected")}
    tot["recall_vs_verified"] = tot["recovered"] / tot["verified_tracks"] \
        if tot["verified_tracks"] else float("nan")
    reviewed_all = bool(df.model_reviewed.all())
    if tot["model_tracks"]:
        if reviewed_all:
            nf = int(df.new_finds.fillna(0).sum())
            tot["new_finds"] = nf
            tot["precision_low"] = tot["precision_high"] = \
                (tot["confirmed"] + nf) / tot["model_tracks"]
        else:
            tot["precision_low"] = tot["confirmed"] / tot["model_tracks"]
            tot["precision_high"] = (tot["model_tracks"] - tot["known_false"]) / tot["model_tracks"]
    df = pd.concat([df, pd.DataFrame([{"clip": "TOTAL", **tot}])], ignore_index=True)

    (work / "eval").mkdir(parents=True, exist_ok=True)
    summary = work / "eval" / ("summary_heldout.csv" if args.heldout else "summary.csv")
    df.to_csv(summary, index=False)

    print()
    if args.heldout:
        print("HELD-OUT FRAMES ONLY (every 4th 32-frame block; never trained on)")
    print(f"{'clip':10s} {'frames':>6s} {'verif':>5s} {'found':>5s} {'recall':>6s} "
          f"{'model':>5s} {'conf':>5s} {'false':>5s} {'only':>5s} {'precision':>13s} {'ball.med':>8s}")
    for r in df.to_dict("records"):
        lo, hi = r.get("precision_low"), r.get("precision_high")
        prec = "  -  " if lo is None or (isinstance(lo, float) and np.isnan(lo)) else (
            f"{lo:.2f}" if abs(lo - hi) < 1e-9 else f"{lo:.2f}-{hi:.2f}")
        print(f"{str(r['clip'])[:10]:10s} {int(r['frames_compared']):6d} "
              f"{int(r['verified_tracks']):5d} {int(r['recovered']):5d} "
              f"{fmt(r['recall_vs_verified']):>6s} {int(r['model_tracks']):5d} "
              f"{int(r['confirmed']):5d} {int(r['known_false']):5d} {int(r['model_only']):5d} "
              f"{prec:>13s} {fmt(r.get('ballistic_median_px')):>8s}")
    print(f"\nwritten: {summary}")

    if not df.classical_reviewed.iloc[:-1].all():
        print("\n!! Some clips have no work/clips/<clip>/reject.txt, so their 'verified'")
        print("   tracks are unreviewed classical tracks -- recall is against those.")
    if not reviewed_all:
        print("\nPrecision is a RANGE until you review the model-only tracks:")
        print("  low  = every model-only track is a false positive")
        print("  high = every model-only track is a real ember the classical detector missed")
        print("Open work/eval/<clip>/model_only/sheet_*.png, write the IDs that are NOT")
        print("firebrands into work/tags/<clip>/reject.txt, and run this again.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
