"""Footage with NO firebrands: which frames, how they are sampled, and the check
that nothing in them moves like an ember.

<work>/negatives.txt lists, per clip, stretches where no firebrand is anywhere
in view, as frame numbers or as times in the video:

    # clip        frames, or times as VLC shows them (m:ss or h:mm:ss)
    clip3         0-150
    clip5         400-520, 900-1010
    quiet_30min   0:00-6:40, 9:15-22:30
    clip8                                     # the whole clip

Clips are looked up in clips/ and negclips/. Put long firebrand-free videos in
negclips/: detect and tag only read clips/, so they never spend hours on
footage that is only there to show "nothing".

How the frames are used -- three things that matter once there is a lot of it:

* SPREAD, NOT FIRST-N. Consecutive frames are near-copies of each other; what
  teaches the model is many different moments (light, smoke, wind, exposure).
  The listed footage is cut into 32-frame blocks and frames are taken from
  blocks spread evenly over all of it, coarse-to-fine, so any budget covers the
  whole video. Each block costs only a seek, not a decode from frame 0.
* HELD OUT. Every 4th block (by position in the clip, so it never changes with
  budgets) is reserved for the eval stage's "false tracks per 100 firebrand-free
  frames" and never trained on. Measured on frames the model was trained on,
  that number would be flattering, and the threshold picked from it too low.
* CHECKED. Every block that is used goes through the classical detector first.
  If anything in it forms a track, the block is dropped and logged with its
  time, so a faint ember you missed while watching does not get taught as
  background. Look at the logged times; if there really is an ember, take that
  stretch out of negatives.txt.
"""
from __future__ import annotations

import math
import multiprocessing as mp
import re
import zlib
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np

from . import detect as D, frames as F, track as T
from .config import DetectConfig, TrackConfig
from .dataset import residual_stacks, tile_positions

BLOCK = 32          # frames per block; the unit of sampling and of the train/eval split
EVAL_EVERY = 4      # every 4th block is held out for the eval stage
MIN_RUN = 8         # shorter stretches are not worth a seek and a background
TIME_PAD_S = 0.25   # trimmed off each end of a stretch given as times: watching
                    # at speed, you cannot place an ember's entry to the frame


# ---------------------------------------------------------------------------
# negatives.txt
# ---------------------------------------------------------------------------

def clock(sec):
    """seconds -> m:ss.s (or h:mm:ss.s), what a video player shows"""
    sec = max(float(sec), 0.0)
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h)}:{int(m):02d}:{s:04.1f}" if h >= 1 else f"{int(m)}:{s:04.1f}"


_TIME = re.compile(r"^(?:(\d+):)?(\d+):(\d+(?:\.\d+)?)$")


def parse_time(tok):
    """'6:40', '6:40.5', '1:02:03' -> seconds; anything else -> None."""
    m = _TIME.match(tok)
    if not m:
        return None
    h, mm, ss = m.groups()
    if float(ss) >= 60 or (h is not None and int(mm) >= 60):
        raise SystemExit(f"negatives.txt: {tok!r} is not a time (seconds and minutes "
                         f"go up to 59)")
    return int(h or 0) * 3600 + int(mm) * 60 + float(ss)


def read_specs(path: Path):
    """negatives.txt -> {clip_stem: [None | ("f", first, last) | ("t", t0, t1)]}.
    None means the whole clip."""
    out = {}
    path = Path(path)
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
            if a.isdigit() and (not sep or b.isdigit()):
                lo, hi = int(a), int(b) if sep else int(a)
                ranges.append(("f", min(lo, hi), max(lo, hi)))
                continue
            ta = parse_time(a)
            tb = parse_time(b) if sep else None
            if ta is not None and tb is not None:
                ranges.append(("t", min(ta, tb), max(ta, tb)))
                continue
            raise SystemExit(f"negatives.txt: cannot read {tok!r} in {line!r} -- use "
                             f"FIRST-LAST frame numbers (120-300) or video times "
                             f"(6:40-12:05)")
        out.setdefault(name, []).extend(ranges or [None])
    return out


@dataclass(frozen=True)
class Span:
    name: str
    path: str
    a: int
    b: int
    fps: float


def resolve(specs, paths, log=print):
    """Specs -> merged frame spans per clip. `paths` maps clip stem -> file."""
    out = []
    for name in sorted(specs):
        path = paths.get(name)
        if path is None:
            log(f"  negatives.txt: no clip called {name} in clips/ or negclips/ -- skipped")
            continue
        src = F.VideoSource(path)
        fps = float(src.fps or 24.0)
        last = (len(src) or 0) - 1
        if last < 0:
            log(f"  negatives.txt: {name}: no frames -- skipped")
            continue
        pad = int(round(TIME_PAD_S * fps))
        spans = []
        for r in specs[name]:
            if r is None:
                a, b = 0, last
            elif r[0] == "f":
                a, b = r[1], r[2]
            else:
                a = int(math.ceil(r[1] * fps)) + pad
                b = int(math.floor(r[2] * fps)) - 1 - pad
            a, b = max(a, 0), min(b, last)
            if b - a + 1 >= MIN_RUN:
                spans.append([a, b])
        spans.sort()
        merged = []
        for a, b in spans:
            if merged and a <= merged[-1][1] + 1:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        out += [Span(name, str(path), a, b, fps) for a, b in merged]
    return out


# ---------------------------------------------------------------------------
# which frames: blocks, train/eval roles, spread order
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Run:
    name: str
    path: str
    a: int
    b: int
    fps: float
    role: str          # "train" or "eval"

    @property
    def n(self):
        return self.b - self.a + 1

    def when(self):
        return f"{clock(self.a / self.fps)}-{clock((self.b + 1) / self.fps)}"

    def label(self):
        return f"{self.name} {self.when()} (frames {self.a}-{self.b})"


def plan(spans, block=BLOCK, eval_every=EVAL_EVERY, min_len=MIN_RUN):
    """Cut the spans into blocks on a fixed grid of the clip's own frame numbers,
    and give each a role. The role depends only on the clip and the block's
    position in it, so tiles and eval -- separate stages, possibly separate jobs,
    possibly different budgets -- always agree on which frames are held out."""
    runs = []
    for s in spans:
        phase = zlib.crc32(s.name.encode()) % eval_every
        for k in range(s.a // block, s.b // block + 1):
            a, b = max(s.a, k * block), min(s.b, k * block + block - 1)
            if b - a + 1 < min_len:
                continue
            role = "eval" if (k + phase) % eval_every == 0 else "train"
            runs.append(Run(s.name, s.path, a, b, s.fps, role))
    return runs


def _vdc(i):
    """van der Corput: 0, .5, .25, .75, .125 ... -- sorting by it orders items
    coarse-to-fine, so every prefix is spread evenly over the whole list."""
    x, f = 0.0, 0.5
    while i:
        x += f * (i & 1)
        i >>= 1
        f /= 2
    return x


def pick(runs, role, budget, run_len=None):
    """Candidate runs for `role`, in the order to use them: every prefix spread
    evenly over all the listed footage. Where there is plenty, each block is cut
    down to its central `run_len` frames -- more moments for the same budget.
    The caller stops when it has `budget` frames it could use."""
    rs = [r for r in runs if r.role == role]
    if not rs or budget <= 0:
        return []
    if run_len and len(rs) * run_len >= budget:
        rs = [r if r.n <= run_len else replace(r, a=r.a + (r.n - run_len) // 2,
                                               b=r.a + (r.n - run_len) // 2 + run_len - 1)
              for r in rs]
    return [rs[i] for i in sorted(range(len(rs)), key=_vdc)]


def describe(runs, log=print):
    """One line on how much footage negatives.txt gives, and the split."""
    if not runs:
        return
    clips = sorted({r.name for r in runs})
    fps = runs[0].fps or 24.0
    tr = sum(r.n for r in runs if r.role == "train")
    ev = sum(r.n for r in runs if r.role == "eval")
    log(f"  negatives.txt: {tr + ev} firebrand-free frames (~{(tr + ev) / fps / 60:.1f} min) "
        f"in {len(clips)} clip(s); {tr} for training, {ev} held out for the eval")


# ---------------------------------------------------------------------------
# reading a run
# ---------------------------------------------------------------------------

def open_at(path, frame):
    """VideoCapture positioned at `frame`. Seeking is exact on this camera's
    HEVC (checked frame by frame); if a file ever reports a different position
    after the seek, fall back to decoding forward from the start."""
    cap = cv2.VideoCapture(str(path))
    if frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        if int(round(cap.get(cv2.CAP_PROP_POS_FRAMES))) != frame:
            cap.release()
            cap = cv2.VideoCapture(str(path))
            for _ in range(frame):
                if not cap.grab():
                    break
    return cap


def read_frames(run, n=None):
    """The raw frames of a run (or its first n): a clean background run."""
    cap = open_at(run.path, run.a)
    out = []
    try:
        for _ in range(run.n if n is None else min(n, run.n)):
            ok, f = cap.read()
            if not ok:
                break
            out.append(f)
    finally:
        cap.release()
    return out


def run_frames(run, dcfg: DetectConfig, roi=None, n_stack=3):
    """Yield (frame_index, residual_stack, edge_map) for every frame of `run`.

    Decoding starts half a median window (plus the stack) early and runs as far
    past the end, so every frame of the run has a full background on both
    sides. The result is what the tag stage computes for that frame, except that
    the stabiliser aligns to the first frame read here rather than the clip's
    first frame (checked: with stabilisation off the stacks are identical). The
    channel follows the run's own first frame, so a clip that switches to IR
    night mode part-way is handled stretch by stretch.
    """
    half = dcfg.median_window // 2
    lo = max(0, run.a - half - n_stack // 2 - 1)
    hi = run.b + half + n_stack // 2
    cap = open_at(run.path, lo)
    ok, first = cap.read()
    if not ok:
        cap.release()
        return
    if F.is_ir(first) and dcfg.channel != "gray":
        dcfg = replace(dcfg, channel="gray")

    def frames():
        try:
            yield first
            for _ in range(lo + 1, hi + 1):
                ok, f = cap.read()
                if not ok:
                    return
                yield f
        finally:
            cap.release()

    for k, stack, edge in residual_stacks(frames(), dcfg, roi, n_stack, with_edge=True):
        fi = lo + k
        if run.a <= fi <= run.b:
            yield fi, stack, edge


def random_tiles(stack, roi, tile, stride, k, rng):
    """Up to k tiles at random grid positions, at least half inside the mask."""
    pos = tile_positions(stack.shape[1:], tile, stride)
    rng.shuffle(pos)
    out = []
    for (y, x) in pos:
        if len(out) >= k:
            break
        if roi is not None and (roi[y:y + tile, x:x + tile] > 0).mean() < 0.5:
            continue
        out.append(stack[:, y:y + tile, x:x + tile])
    return out


def hard_tiles(dets, stack, roi, tile, k, rng):
    """Up to k tiles around the places a model fired on footage with nothing in
    it -- the texture it mistakes for a firebrand, which is exactly what it has
    to learn is not one. Strongest first, one tile per place, the place off
    centre by a random amount so position in the tile carries no signal."""
    H, W = stack.shape[1:]
    out, taken = [], []
    for d in sorted(dets, key=lambda d: -d.peak):
        if len(out) >= k:
            break
        if any(abs(d.x - x) < tile / 4 and abs(d.y - y) < tile / 4 for x, y in taken):
            continue
        jx, jy = rng.integers(-tile // 4, tile // 4 + 1, 2)
        x0 = int(np.clip(round(d.x) - tile // 2 + jx, 0, max(W - tile, 0)))
        y0 = int(np.clip(round(d.y) - tile // 2 + jy, 0, max(H - tile, 0)))
        if roi is not None and (roi[y0:y0 + tile, x0:x0 + tile] > 0).mean() < 0.5:
            continue
        out.append(stack[:, y0:y0 + tile, x0:x0 + tile])
        taken.append((d.x, d.y))
    return out


def moving_tracks(dets_by_frame, tcfg: TrackConfig):
    """What the classical detector would accept as firebrand tracks here."""
    if not any(dets_by_frame.values()):
        return []
    return T.accept(T.link(dets_by_frame, tcfg, step_ratio=T.resolve_step_ratio(tcfg)), tcfg)


def describe_tracks(tracks, run, limit=3):
    bits = []
    for t in tracks[:limit]:
        d = t.dets[0]
        bits.append(f"{clock(d.frame / run.fps)} at x={d.x:.0f} y={d.y:.0f} "
                    f"({len(t.dets)} frames)")
    more = f" and {len(tracks) - limit} more" if len(tracks) > limit else ""
    return "; ".join(bits) + more


# ---------------------------------------------------------------------------
# harvesting training runs (in worker processes)
# ---------------------------------------------------------------------------

@dataclass
class Job:
    run: Run
    dcfg: DetectConfig
    tcfg: TrackConfig          # step_ratio set to the camera's
    roi: object                # uint8 mask or None
    n_stack: int
    tile: int
    stride: int
    rand_per_frame: int
    keep_every: int            # keep every n-th stack for hard-negative mining; 0 = none


def harvest(job: Job):
    """One training run -> random tiles, a few full stacks for mining, and the
    classical check. A run where anything forms a track gives nothing."""
    cv2.setNumThreads(2)
    run = job.run
    rng = np.random.default_rng(zlib.crc32(f"{run.name}:{run.a}".encode()))
    centre = job.n_stack // 2
    dets, tiles, kept, n = {}, [], [], 0
    for j, (fi, stack, edge) in enumerate(run_frames(run, job.dcfg, job.roi, job.n_stack)):
        n += 1
        dets[fi] = D.detect_frame(stack[centre], fi, job.dcfg, edge)
        tiles += random_tiles(stack, job.roi, job.tile, job.stride, job.rand_per_frame, rng)
        if job.keep_every and j % job.keep_every == job.keep_every // 2:
            kept.append((fi, stack))
    moving = moving_tracks(dets, job.tcfg)
    if moving:
        return dict(run=run, n=n, moving=describe_tracks(moving, run), n_moving=len(moving),
                    tiles=None, stacks=[])
    shape = (0, job.n_stack, job.tile, job.tile)
    return dict(run=run, n=n, moving=None, n_moving=0,
                tiles=np.array(tiles, np.uint8) if tiles else np.zeros(shape, np.uint8),
                stacks=kept)


def default_workers():
    import os
    return max(1, min(4, (os.cpu_count() or 2) // 2))


def harvest_runs(jobs, budget, workers=1, log=print):
    """Yield harvest() results in job order until `budget` usable frames are in
    hand. Runs are independent, so they go to `workers` processes (spawned, not
    forked: the parent may already hold a CUDA context)."""
    from concurrent.futures.process import BrokenProcessPool

    jobs = iter(jobs)
    kept = 0

    def serial(todo):
        nonlocal kept
        for job in todo:
            r = harvest(job)
            yield r
            if not r["moving"]:
                kept += r["n"]
            if kept >= budget:
                return

    if workers <= 1:
        yield from serial(jobs)
        return
    try:
        ex = ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"))
    except (OSError, ValueError) as e:
        log(f"  (no worker processes: {e} -- reading runs one at a time)")
        yield from serial(jobs)
        return
    pending = deque()            # (job, future), in job order
    broken = None
    try:
        def fill():
            while len(pending) < 2 * workers:
                job = next(jobs, None)
                if job is None:
                    return
                pending.append((job, ex.submit(harvest, job)))

        fill()
        while pending:
            job, fut = pending[0]
            try:
                r = fut.result()
            except BrokenProcessPool as e:
                broken = e
                break
            pending.popleft()
            yield r
            if not r["moving"]:
                kept += r["n"]
            if kept >= budget:
                return
            fill()
    finally:
        for _job, f in pending:
            f.cancel()
        ex.shutdown(wait=True, cancel_futures=True)
    if broken is not None:
        # A worker died (out of memory, or the container forbids processes).
        # Nothing is lost: redo the unfinished runs here, one at a time.
        log(f"  (worker processes failed: {broken} -- reading the rest one at a time)")
        todo = [job for job, _f in pending]
        pending.clear()
        yield from serial(_chain(todo, jobs))


def _chain(first, rest):
    yield from first
    yield from rest
