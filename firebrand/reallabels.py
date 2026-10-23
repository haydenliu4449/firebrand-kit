"""Real firebrands as training labels: your verified classical tracks and the
model's own tags.

Until now every positive the model trained on was a synthetic ember. Real
embers differ from the generator's (shape, glow, compression, motion blur),
and the only way to teach those differences is to train on real ones. You
already have them:

  * VERIFIED CLASSICAL TRACKS -- detect's accepted tracks minus the IDs in
    work/clips/<clip>/reject.txt. Reliable, but only the embers the classical
    detector can see.
  * THE MODEL'S TAGS -- work/tags/<clip>/detections.csv, minus the IDs in
    work/tags/<clip>/reject.txt (written after reviewing evaluate_tags.py's
    model-only sheets), and only tracks where the model was confident
    (--real-min-conf). These include embers the classical detector missed,
    which is how each round of training can find more than the last.

The catch, and how it is handled: a real frame always holds embers nobody
labelled -- too faint for the classical detector, below the model's
threshold, or simply missed. Train on the frame as it is and every one of them
is taught as "not a firebrand", which pushes recall DOWN. So in the frame being
labelled (the centre of the 3-frame stack), every ember-like blob that carries
no label is removed: set to zero, which is what the residual is everywhere
there is nothing. Specks too small or dim to be an ember stay, so the camera's
grain looks as it does when tagging. Firebrand-free footage
(negatives.txt) is what teaches it what is NOT a firebrand.

Frames in the held-out blocks (negatives.is_heldout: every 4th 32-frame block)
are never used, so `python evaluate_tags.py --heldout` measures the next model
on frames it has not trained on.
"""
from __future__ import annotations

import pickle
import zlib
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from . import detect as D
from .config import DetectConfig
from .negatives import BLOCK, MIN_RUN, Run, is_heldout, run_frames


def _ids(path: Path):
    if not path.exists():
        return set()
    txt = path.read_text().replace(",", " ").split()
    return {int(t) for t in txt if t.strip().lstrip("-").isdigit()}


def read_csv_or_empty(path: Path):
    """A CSV the kit wrote, or an empty table. A clip with no tracks used to
    get a file holding only a newline, which pandas will not read."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _rows(dets):
    return np.array([[d.x, d.y, d.L, d.W, d.theta] for d in dets], np.float32).reshape(-1, 5)


def review_is_current(work, name):
    """True if work/tags/<clip>/reject.txt was written after the clip was last
    tagged. An older one lists IDs of tracks that no longer exist -- re-tagging
    renumbers them -- and would reject the wrong ones."""
    d = Path(work) / "tags" / name
    rej, done = d / "reject.txt", d / "done.json"
    return rej.exists() and (not done.exists() or rej.stat().st_mtime >= done.stat().st_mtime)


def clip_labels(work, name, min_conf=0.8, max_w=12.0, dedupe_px=10.0):
    """{frame: (n, 5) array of x, y, L, W, theta} of firebrands to train on,
    plus counts. A detection too wide to be one streak (max_w, px; neighbours
    merged at the classical threshold) is not a label -- it is removed from
    the frame like any other unlabelled blob."""
    work = Path(work)
    per = {}
    n_cls = n_mod = n_wide = n_dup = 0

    pk = work / "clips" / name / "tracks.pkl"
    if pk.exists() and (work / "clips" / name / "done.json").exists():
        rej = _ids(work / "clips" / name / "reject.txt")
        with open(pk, "rb") as fh:
            acc = pickle.load(fh)["accepted"]
        for t in acc:
            if t.id in rej:
                continue
            for d in t.dets:
                per.setdefault(d.frame, {"c": [], "m": []})["c"].append(d)

    df = read_csv_or_empty(work / "tags" / name / "detections.csv")
    if len(df) and "track_id" in df:         # a clip the model tagged nothing in has none
        rej = _ids(work / "tags" / name / "reject.txt") \
            if review_is_current(work, name) else set()
        conf = df.groupby("track_id")["peak"].max() >= 255.0 * min_conf
        ok = set(conf[conf].index.astype(int)) - rej
        df = df[df.track_id.astype(int).isin(ok)]
        for r in df.itertuples():
            per.setdefault(int(r.frame), {"c": [], "m": []})["m"].append(r)

    out = {}
    for f, g in per.items():
        m = [(r.x, r.y, r.L, r.W, r.theta) for r in g["m"]]
        c = []
        for d in g["c"]:
            if m and min((d.x - x) ** 2 + (d.y - y) ** 2 for x, y, *_ in m) < dedupe_px ** 2:
                n_dup += 1                      # the same ember: keep the model's thinner fit
                continue
            c.append((d.x, d.y, d.L, d.W, d.theta))
        rows = []
        for x, y, L, W, th in m + c:
            if W > max_w:
                n_wide += 1
                continue
            rows.append((x, y, L, W, th))
        n_mod += len(m)
        n_cls += len(c)
        if rows:
            out[f] = np.array(rows, np.float32)
    return out, dict(classical=n_cls, model=n_mod, both=n_dup, too_wide=n_wide)


def label_runs(name, path, fps, labels, n_frames, block=BLOCK):
    """Training blocks of a clip that contain labels (held-out blocks excluded)."""
    runs = []
    if not labels:
        return runs
    last = max(min(n_frames - 1, max(labels)), 0)
    for k in range(min(labels) // block, last // block + 1):
        a, b = k * block, min(k * block + block - 1, last)
        if b - a + 1 < MIN_RUN or is_heldout(name, a):
            continue
        if any(f in labels for f in range(a, b + 1)):
            runs.append(Run(name, str(path), a, b, fps, "train"))
    return runs


class _Det:                                     # what detect.box_points needs
    __slots__ = ("x", "y", "L", "W", "theta")

    def __init__(self, x, y, L, W, theta):
        self.x, self.y, self.L, self.W, self.theta = x, y, L, W, theta


def draw_targets(rows, x0, y0, tile, width=2.0):
    """The training target, drawn exactly like synth.gt_masks: a 2 px line
    along each streak's core. A measured streak is its core plus the blur on
    both ends, so the core length is L - W."""
    m = np.zeros((tile, tile), np.float32)
    for x, y, L, W, th in rows:
        core = max(float(L) - float(W), 1.0)
        dx, dy = np.cos(th) * core / 2, np.sin(th) * core / 2
        cx, cy = x - x0, y - y0
        cv2.line(m, (int(round(cx - dx)), int(round(cy - dy))),
                 (int(round(cx + dx)), int(round(cy + dy))),
                 1.0, thickness=max(int(round(width)), 1), lineType=cv2.LINE_AA)
    return np.clip(m, 0, 1)


def clean_centre(cen, rows, x0, y0, thr, peak_thr, min_area=2, pad=4.0):
    """Zero everything ember-like in the centre frame that is not a labelled
    firebrand.

    Kept: each label's own box, `pad` px around it (the streak and its glow).
    Removed: every bright core outside those boxes -- at least `min_area`
    pixels above `peak_thr` (median + 4.5 sigma) -- with its glow, i.e. whatever
    is above `thr` within 4 px. The decision is per pixel, not per blob: in a
    dense shower a bright ember's glow merges with the labelled streaks next to
    it, and judging the merged blob as a whole would keep the unlabelled ember.
    Specks too small or dim to be an ember stay, so the model sees the same
    grain it sees when tagging; a tile scrubbed of all noise would teach it
    that any speck left must be a firebrand. Returns the number of cores
    removed."""
    keep = np.zeros(cen.shape, np.uint8)
    for x, y, L, W, th in rows:
        p = D.box_points(_Det(x - x0, y - y0, L, W, th), pad=pad)
        cv2.fillPoly(keep, [np.round(p).astype(np.int32)], 1)
    outside = ((cen > peak_thr) & (keep == 0)).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(outside, connectivity=8)
    if n <= 1:
        return 0
    big = stats[:, cv2.CC_STAT_AREA] >= min_area
    big[0] = False
    gone = np.where(big)[0]
    if len(gone):
        g = np.isin(lab, gone).astype(np.uint8)
        glow = (cv2.dilate(g, np.ones((9, 9), np.uint8)) > 0) & (cen > thr)
        drop = (glow | (cv2.dilate(g, np.ones((3, 3), np.uint8)) > 0)) & (keep == 0)
        cen[drop] = 0
    return int(len(gone))


@dataclass
class PosJob:
    run: Run
    dcfg: DetectConfig
    roi: object
    n_stack: int
    tile: int
    labels: dict               # frame -> (n, 5) rows, frames of this run only
    per_frame: int = 6
    blob_k: float = 3.0        # a blob's extent: pixels above median + blob_k * sigma
    blob_peak_k: float = 4.5   # ...and it is ember-like if its peak reaches median + this * sigma
    blob_min_area: int = 2     # ...over at least this many pixels


def harvest_pos(job: PosJob):
    """One block of a real clip -> tiles centred on labelled firebrands, the
    centre frame cleaned of unlabelled blobs, targets drawn on the labels."""
    cv2.setNumThreads(2)
    run, t, c = job.run, job.tile, job.n_stack // 2
    rng = np.random.default_rng(zlib.crc32(f"pos:{run.name}:{run.a}".encode()))
    X, Y = [], []
    n = n_lab = n_removed = 0
    for fi, stack, _edge in run_frames(run, job.dcfg, job.roi, job.n_stack):
        rows = job.labels.get(fi)
        if rows is None or not len(rows):
            continue
        H, W = stack.shape[1:]
        cen_full = stack[c]
        nz = cen_full[cen_full > 0]
        base = float(np.median(nz)) if nz.size else 0.0
        sigma = D.robust_sigma(cen_full)
        thr, peak_thr = base + job.blob_k * sigma, base + job.blob_peak_k * sigma
        took, centres = 0, []
        for i in rng.permutation(len(rows)):
            if took >= job.per_frame:
                break
            x, y = float(rows[i, 0]), float(rows[i, 1])
            if any(abs(x - a) < t / 4 and abs(y - b) < t / 4 for a, b in centres):
                continue
            jx, jy = rng.integers(-t // 4, t // 4 + 1, 2)
            x0 = int(np.clip(round(x) - t // 2 + jx, 0, max(W - t, 0)))
            y0 = int(np.clip(round(y) - t // 2 + jy, 0, max(H - t, 0)))
            if job.roi is not None and (job.roi[y0:y0 + t, x0:x0 + t] > 0).mean() < 0.5:
                continue
            near = rows[(rows[:, 0] > x0 - t / 4) & (rows[:, 0] < x0 + t * 1.25) &
                        (rows[:, 1] > y0 - t / 4) & (rows[:, 1] < y0 + t * 1.25)]
            xs = stack[:, y0:y0 + t, x0:x0 + t].copy()
            n_removed += clean_centre(xs[c], near, x0, y0, thr, peak_thr, job.blob_min_area)
            ys = draw_targets(near, x0, y0, t)
            if ys.max() <= 0:
                continue
            X.append(xs)
            Y.append(ys)
            centres.append((x, y))
            took += 1
            n_lab += int(((near[:, 0] >= x0) & (near[:, 0] < x0 + t) &
                          (near[:, 1] >= y0) & (near[:, 1] < y0 + t)).sum())
        if took:
            n += 1
    shape = (0, job.n_stack, t, t)
    return dict(run=run, n=n, moving=None, n_labels=n_lab, removed=n_removed,
                X=np.array(X, np.uint8) if X else np.zeros(shape, np.uint8),
                Y=np.array(Y, np.float32) if Y else np.zeros((0, t, t), np.float32))
