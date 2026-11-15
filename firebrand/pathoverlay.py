"""Whole paths on the overlay video: everywhere a firebrand has been, joined up.

The plain overlay boxes a firebrand only on the frames it is detected in, so its
path has to be followed by eye from one box to the next. Here every position an
ember has been tagged at stays on screen, joined in frame order into one line in
the ember's colour, with a dot where it was on each frame (the spacing of the
dots is its speed). An "ember" is whatever ID the tags carry: a track, or a
trajectory -- the tracks of one ember across its bounces (trajectory.py). So this
draws what the tracker says, no more: a path that jumps between two embers is the
tracker's mistake made visible (approach.md, finding 5), and the picture improves
as the tracker does.

How it is drawn:
  * A position joins the line on the NEXT frame. On the frame an ember is
    detected in, its line stops at the edge of its box: nothing is ever drawn
    over the firebrand you are looking at, the same rule as detect.draw.
  * A path stays until the end of the clip, or `linger` frames after the ember
    was last seen.
  * Colours repeat; the number beside the box is the ID.

Pure OpenCV and numpy, no model: runs wherever a clip and its detections are.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# BGR. Bright against the fire-lit scene, and none of them the overlay's box green.
PALETTE = [(255, 0, 255), (255, 200, 0), (255, 255, 255), (180, 105, 255),
           (0, 165, 255), (255, 255, 0), (255, 96, 128), (128, 255, 255),
           (200, 255, 160), (128, 128, 255), (255, 160, 200), (0, 230, 255)]

COLS = ["x", "y", "L", "W", "theta"]


def colour(ember_id):
    return PALETTE[int(ember_id) % len(PALETTE)]


@dataclass
class EmberPath:
    id: int
    frames: np.ndarray      # (n,) frame numbers, rising
    rows: np.ndarray        # (n, 5): x, y, L, W, theta of the detection on each


def from_tracks(tracks, group=None):
    """Tracks (.id, .dets) -> one path per ember.

    `group` maps a track id to its ember's id (trajectory.build's traj_of): the
    tracks of one ember then make ONE path, joined across each bounce. Without
    it every track is its own path."""
    by = {}
    for t in tracks:
        by.setdefault(int(t.id if group is None else group[t.id]), []).extend(t.dets)
    out = []
    for pid in sorted(by):
        dets = sorted(by[pid], key=lambda d: d.frame)
        out.append(EmberPath(pid, np.array([d.frame for d in dets], int),
                             np.array([[d.x, d.y, d.L, d.W, d.theta] for d in dets], float)))
    return out


def id_column(df, by="auto"):
    """The column that says which ember a row belongs to.

    by: "track" (track_id), "trajectory" (trajectory_id: one ember across its
    hops) or "auto", which takes trajectories where the table has them."""
    if by in ("auto", "trajectory"):
        if "trajectory_id" in df and df["trajectory_id"].notna().all():
            return "trajectory_id"
        if by == "trajectory":
            raise ValueError("no trajectory_id in this table: run trajectories.py on the "
                             "tags first, or draw by track")
    return "track_id"


def from_table(df, by="auto"):
    """Rows of a detections.csv (or a hand-tag CSV: same columns) -> paths, one
    per ember (see id_column for `by`)."""
    if not len(df) or "track_id" not in df:
        return []
    col = id_column(df, by)
    df = df.sort_values([col, "frame"], kind="stable")
    return [EmberPath(int(pid), g["frame"].to_numpy(int), g[COLS].to_numpy(float))
            for pid, g in df.groupby(col, sort=True)]


def _q(xy):
    """A point for OpenCV's shift=2 drawing: quarter pixels, so a slow path does
    not come out as a staircase."""
    return int(round(float(xy[0]) * 4)), int(round(float(xy[1]) * 4))


class PathOverlay:
    """draw(frame, i) -> a copy of the frame with every path as far as frame i.

    Call it with rising frame numbers: it keeps what it has drawn and only adds
    to it. A first call at frame 200 shows the paths as they stood by then.
    Sizes follow the frame width (px at 1920 wide, like every pixel size here).
    """

    def __init__(self, paths, shape, linger=None, labels=True, thickness=None, box_pad=4.0):
        H, W = shape[:2]
        scale = max(1.0, W / 1920.0)
        self.paths = [p for p in paths if len(p.frames)]
        self.linger = None if linger is None else max(int(linger), 0)
        self.labels, self.pad = bool(labels), float(box_pad)
        self.th = int(thickness or max(1, round(scale)))
        self.font = 0.4 * scale
        self.layer = np.zeros((H, W, 3), np.uint8)      # the lines: colour x coverage
        self.alpha = np.zeros((H, W), np.uint8)         # coverage, 0..255
        self.box = None                                 # x0, y0, x1, y1 of what is on them
        self.events = sorted((int(f), j, k) for j, p in enumerate(self.paths)
                             for k, f in enumerate(p.frames))
        self.at = {}                                    # frame -> [(path, point)] seen on it
        for f, j, k in self.events:
            self.at.setdefault(f, []).append((j, k))
        self.done = 0                                   # events already on the layer
        self.first = np.array([p.frames[0] for p in self.paths], int)
        self.last = np.array([p.frames[-1] for p in self.paths], int)

    # -- the kept layer ------------------------------------------------------
    def _put(self, j, k):
        """Point k of path j onto the layer: its dot, and the line from point k-1."""
        p = self.paths[j]
        col, b = colour(p.id), p.rows[k, :2]
        pts = [b]
        if k:
            a = p.rows[k - 1, :2]
            pts.append(a)
            for img, c in ((self.layer, col), (self.alpha, 255)):
                cv2.line(img, _q(a), _q(b), c, self.th, cv2.LINE_AA, shift=2)
        for img, c in ((self.layer, col), (self.alpha, 255)):
            cv2.circle(img, _q(b), (self.th + 1) * 4, c, -1, cv2.LINE_AA, shift=2)
        m = self.th + 4
        x0, y0 = np.floor(np.min(pts, 0)) - m
        x1, y1 = np.ceil(np.max(pts, 0)) + m + 1
        o = self.box
        self.box = (x0, y0, x1, y1) if o is None else \
            (min(o[0], x0), min(o[1], y0), max(o[2], x1), max(o[3], y1))

    def _region(self):
        if self.box is None:
            return None
        H, W = self.alpha.shape
        x0, y0 = int(max(self.box[0], 0)), int(max(self.box[1], 0))
        x1, y1 = int(min(self.box[2], W)), int(min(self.box[3], H))
        return (slice(y0, y1), slice(x0, x1)) if x1 > x0 and y1 > y0 else None

    def _advance(self, i):
        """Bring the layer up to frame i: every position from an earlier frame."""
        if self.linger is None:                         # nothing ever leaves: add the new
            while self.done < len(self.events) and self.events[self.done][0] < i:
                self._put(*self.events[self.done][1:])
                self.done += 1
            return
        r = self._region()                              # paths leave: draw afresh
        if r is not None:
            self.layer[r] = 0
            self.alpha[r] = 0
        self.box = None
        for j in np.nonzero((self.first < i) & (self.last + self.linger >= i))[0]:
            for k in range(int(np.searchsorted(self.paths[j].frames, i))):
                self._put(int(j), k)

    # -- one frame -----------------------------------------------------------
    def _reach(self, row, u):
        """From a detection's centre to the edge of its drawn box, going along u."""
        _x, _y, L, W, th = row
        a, b = (L + 1) / 2 + self.pad, (W + 1) / 2 + self.pad
        along = abs(u[0] * np.cos(th) + u[1] * np.sin(th))
        across = abs(u[1] * np.cos(th) - u[0] * np.sin(th))
        return 1.0 / max(along / a, across / b, 1e-9)

    def _label_at(self, row):
        x, y, L, W, th = row
        c, s = abs(np.cos(th)), abs(np.sin(th))
        l, w = L + 1 + 2 * self.pad, W + 1 + 2 * self.pad
        return int(x + (c * l + s * w) / 2 + 3), int(y - (s * l + c * w) / 2)

    def draw(self, frame, i):
        i = int(i)
        self._advance(i)
        out = frame.copy()
        r = self._region()
        if r is not None:
            # layer over frame. The layer already holds colour x coverage, so the
            # frame only has to give way by the coverage: frame x (1 - a) + layer.
            keep = cv2.merge([cv2.bitwise_not(self.alpha[r])] * 3)
            out[r] = cv2.add(cv2.multiply(out[r], keep, scale=1 / 255.0), self.layer[r])
        for j, k in self.at.get(i, ()):                 # the embers in view on this frame
            p = self.paths[j]
            col, row = colour(p.id), p.rows[k]
            if k:
                a, v = p.rows[k - 1, :2], row[:2] - p.rows[k - 1, :2]
                dist = float(np.hypot(*v))
                if dist > 1e-6:
                    u = v / dist
                    stop = dist - self._reach(row, u) - self.th
                    if stop > 0:
                        cv2.line(out, _q(a), _q(a + u * stop), col, self.th, cv2.LINE_AA, shift=2)
            if self.labels:
                org = self._label_at(row)
                for c, t in (((0, 0, 0), self.th + 2), (col, self.th)):
                    cv2.putText(out, str(p.id), org, cv2.FONT_HERSHEY_SIMPLEX, self.font, c, t,
                                cv2.LINE_AA)
        return out


def still(frame, paths, upto=None, thickness=None):
    """One picture of every path, whole (or as far as frame `upto`, inclusive)."""
    return PathOverlay(paths, frame.shape, labels=False, thickness=thickness) \
        .draw(frame, 10 ** 9 if upto is None else int(upto) + 1)
