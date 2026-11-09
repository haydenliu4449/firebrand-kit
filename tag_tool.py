#!/usr/bin/env python3
"""Tag firebrands by hand: tight rotated rectangles, one track per ember.

    pip install opencv-python numpy          # on your laptop; Cloud Shell has no screen
    python3 tag_tool.py clip5.mp4            # writes clip5_manual.csv beside the video
    python3 tag_tool.py clip5.mp4 --heldout  # only the frames no model trains on

DRAW
  left-drag   a rough box round an ember: the tool fits the tightest rotated
              rectangle to it (half-maximum outline of the ember in the blue
              channel). Too faint to fit -> your box is kept as drawn (shown dashed).
  left-click  on a box: that ember becomes the current one (continue its track)
  right-click on a box: delete it
  n           new ember (next boxes start a new track)
  u           undo the last box          r   toggle fitting (off = keep boxes as drawn)

MOVE
  d / .       next frame                 a / ,   previous frame
  ]  [        10 frames on / back        h       next held-out frame
  + / =  -    zoom in / out (or the mouse wheel, where your system sends it)
  i j k l     pan up / left / down / right           0   whole frame
  m           view: normal -> enhanced -> motion (frame minus its neighbours:
              moving embers stand out, still ground does not)
  g           ghost boxes from the previous frame on/off
  ?           help on/off                q / Esc save and quit

Every change is saved at once to the CSV (written to a temporary file, then
renamed, so a crash never leaves half a file). Run it again on the same video to
carry on where you stopped. Columns: frame, track_id, x, y, L, W, theta (radians,
the kit's convention: firebrand/detect.py), fitted, x_min..y_max (the upright box
round it), peak, area. The kit reads x, y, L, W, theta, track_id like its own tags.
"""
from __future__ import annotations

import argparse
import csv
import os
import zlib
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np

BLOCK, EVAL_EVERY = 32, 4            # the kit's held-out split (firebrand/negatives.py)
FIELDS = ["frame", "track_id", "x", "y", "L", "W", "theta", "fitted",
          "x_min", "y_min", "x_max", "y_max", "peak", "area"]


def is_heldout(name, frame, block=BLOCK, eval_every=EVAL_EVERY):
    """Same rule as firebrand.negatives.is_heldout (tests/test_tag_tool.py checks)."""
    phase = zlib.crc32(name.encode()) % eval_every
    return (int(frame) // block + phase) % eval_every == 0


# ---------------------------------------------------------------------------
# fitting a box to an ember
# ---------------------------------------------------------------------------

def fit_box(bgr, x0, y0, x1, y1, min_contrast=8.0):
    """Tightest rotated rectangle round the ember inside the box (x0, y0)-(x1, y1).

    Background = median of a 3 px ring just outside the box; the ember = the
    connected pixels at or above half its peak brightness above that background
    (blue channel: embers are near white, the fire-lit scene is orange). Returns
    dict(x, y, L, W, theta, peak, area) or None when there is no clear ember.
    L and W are pixel extents (minAreaRect measures between pixel centres, + 1).
    """
    h, w = bgr.shape[:2]
    x0, x1 = sorted((int(round(x0)), int(round(x1))))
    y0, y1 = sorted((int(round(y0)), int(round(y1))))
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, w - 1), min(y1, h - 1)
    if x1 - x0 < 1 or y1 - y0 < 1:
        return None
    m = 3
    X0, Y0, X1, Y1 = max(x0 - m, 0), max(y0 - m, 0), min(x1 + m, w - 1), min(y1 + m, h - 1)
    b = bgr[Y0:Y1 + 1, X0:X1 + 1, 0].astype(np.float32)
    inner = np.zeros(b.shape, bool)
    inner[y0 - Y0:y1 - Y0 + 1, x0 - X0:x1 - X0 + 1] = True
    ring = b[~inner]
    bg = float(np.median(ring)) if ring.size else float(np.median(b))
    vals = np.where(inner, b, -1)
    py, px = np.unravel_index(int(np.argmax(vals)), vals.shape)
    peak = float(b[py, px]) - bg
    if peak < min_contrast:
        return None
    mask = ((b >= bg + 0.5 * peak) & inner).astype(np.uint8)
    n, lab = cv2.connectedComponents(mask, connectivity=8)
    ys, xs = np.nonzero(lab == lab[py, px])
    if len(xs) < 2:
        return None
    pts = np.stack([xs + X0, ys + Y0], 1).astype(np.float32)
    (cx, cy), (rw, rh), ang = cv2.minAreaRect(pts)
    L, W = max(rw, rh) + 1.0, min(rw, rh) + 1.0
    theta = np.deg2rad(ang if rw >= rh else ang + 90.0)
    theta = float((theta + np.pi / 2) % np.pi - np.pi / 2)      # [-pi/2, pi/2)
    return dict(x=float(cx), y=float(cy), L=float(L), W=float(W), theta=theta,
                peak=round(peak, 1), area=int(len(xs)))


def raw_box(x0, y0, x1, y1):
    """A drawn box kept as it is (upright)."""
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    w, h = x1 - x0 + 1, y1 - y0 + 1
    return dict(x=(x0 + x1) / 2, y=(y0 + y1) / 2, L=float(max(w, h)), W=float(min(w, h)),
                theta=0.0 if w >= h else float(np.pi / 2), peak=0.0, area=0)


def corners(r):
    """The four corners of a box record, (4, 2) image pixels."""
    return cv2.boxPoints(((r["x"], r["y"]), (r["L"], r["W"]), float(np.rad2deg(r["theta"]))))


def with_bounds(r):
    p = corners(r)
    r["x_min"], r["y_min"] = float(p[:, 0].min()), float(p[:, 1].min())
    r["x_max"], r["y_max"] = float(p[:, 0].max()), float(p[:, 1].max())
    return r


# ---------------------------------------------------------------------------
# the CSV
# ---------------------------------------------------------------------------

def load(path):
    if not Path(path).exists():
        return []
    with open(path, newline="") as fh:
        out = []
        for row in csv.DictReader(fh):
            r = {k: float(v) for k, v in row.items() if k in FIELDS and v != ""}
            r["frame"], r["track_id"], r["fitted"] = int(r["frame"]), int(r["track_id"]), int(r.get("fitted", 1))
            r.setdefault("area", 0)
            r["area"] = int(r["area"])
            out.append(r)
        return out


def save(path, boxes):
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for r in sorted(boxes, key=lambda r: (r["frame"], r["track_id"])):
            w.writerow({k: (round(r[k], 3) if isinstance(r[k], float) else r[k]) for k in FIELDS})
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# frames
# ---------------------------------------------------------------------------

class Frames:
    """Frames by index, with a small cache: stepping forward reads on, a jump seeks."""

    def __init__(self, path, keep=48):
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise SystemExit(f"cannot open {path}")
        self.n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.pos = 0
        self.cache: OrderedDict = OrderedDict()
        self.keep = keep

    def get(self, i):
        i = int(i)
        if i in self.cache:
            self.cache.move_to_end(i)
            return self.cache[i]
        if i != self.pos:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            self.pos = i
        ok, f = self.cap.read()
        if not ok:
            return None
        self.pos = i + 1
        self.cache[i] = f
        while len(self.cache) > self.keep:
            self.cache.popitem(last=False)
        return f


# ---------------------------------------------------------------------------
# the window
# ---------------------------------------------------------------------------

def wheel_delta(flags):
    """Mouse-wheel step from an OpenCV mouse callback's flags: > 0 up, < 0 down.

    The C++ helper cv::getMouseWheelDelta is missing from most Python builds of
    OpenCV; the step is the signed high 16 bits of flags (e.g. +/-120 per notch).
    """
    d = (int(flags) >> 16) & 0xFFFF
    return d - 0x10000 if d & 0x8000 else d


class Tagger:
    VIEWS = ("normal", "enhanced", "motion")
    COLS = [(0, 255, 0), (255, 200, 0), (0, 200, 255), (255, 0, 255), (0, 255, 255),
            (255, 128, 0), (128, 255, 128), (200, 120, 255)]

    def __init__(self, video, out, heldout=False, start=0, view=(1600, 900)):
        self.video, self.out, self.only_heldout = Path(video), Path(out), heldout
        self.name = self.video.stem
        self.frames = Frames(video)
        self.boxes = load(out)
        self.track = max((r["track_id"] for r in self.boxes), default=-1) + 1
        self.vw, self.vh = view
        self.f = self._ok_frame(start, +1)
        first = self.frames.get(self.f)
        self.H, self.W = first.shape[:2]
        self.fit_whole()
        self.view, self.ghost, self.help, self.fitting = 0, True, True, True
        self.drag = None
        self.history = []                 # for undo: boxes added, in order
        self.msg = ""

    # -- geometry: image <-> screen -------------------------------------------
    def fit_whole(self):
        self.zoom = min(self.vw / self.W, self.vh / self.H)
        self.cx, self.cy = self.W / 2, self.H / 2

    def to_img(self, sx, sy):
        return self.cx + (sx - self.vw / 2) / self.zoom, self.cy + (sy - self.vh / 2) / self.zoom

    def to_scr(self, x, y):
        return (x - self.cx) * self.zoom + self.vw / 2, (y - self.cy) * self.zoom + self.vh / 2

    def zoom_at(self, sx, sy, factor):
        ix, iy = self.to_img(sx, sy)
        self.zoom = float(np.clip(self.zoom * factor, 0.1, 40.0))
        self.cx, self.cy = ix - (sx - self.vw / 2) / self.zoom, iy - (sy - self.vh / 2) / self.zoom

    # -- frames ------------------------------------------------------------------
    def _ok_frame(self, i, step):
        i = int(np.clip(i, 0, max(self.frames.n - 1, 0)))
        if not self.only_heldout:
            return i
        j = i
        while 0 <= j < self.frames.n and not is_heldout(self.name, j):
            j += 1 if step > 0 else -1
        return j if 0 <= j < self.frames.n else self.f if hasattr(self, "f") else i

    def go(self, i, step):
        j = self._ok_frame(i, step)
        if self.frames.get(j) is not None:
            self.f = j

    def next_heldout(self):
        j = self.f + 1
        while j < self.frames.n and not is_heldout(self.name, j):
            j += 1
        if j < self.frames.n:
            self.f = j

    # -- picture -----------------------------------------------------------------
    def base_image(self):
        f = self.frames.get(self.f)
        v = self.VIEWS[self.view]
        if v == "enhanced":
            lab = cv2.cvtColor(f, cv2.COLOR_BGR2LAB)
            lab[..., 0] = cv2.createCLAHE(3.0, (16, 16)).apply(lab[..., 0])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        if v == "motion":
            a = self.frames.get(max(self.f - 1, 0))
            b = self.frames.get(min(self.f + 1, self.frames.n - 1))
            if a is None or b is None:
                return f
            d = f[..., 0].astype(np.int16) - np.maximum(a[..., 0], b[..., 0]).astype(np.int16)
            d = np.clip(d * 4, 0, 255).astype(np.uint8)
            return cv2.cvtColor(d, cv2.COLOR_GRAY2BGR)
        return f

    def render(self):
        img = self.base_image()
        M = np.float32([[self.zoom, 0, self.vw / 2 - self.cx * self.zoom],
                        [0, self.zoom, self.vh / 2 - self.cy * self.zoom]])
        interp = cv2.INTER_NEAREST if self.zoom >= 2 else cv2.INTER_AREA
        can = cv2.warpAffine(img, M, (self.vw, self.vh), flags=interp, borderValue=(40, 40, 40))
        if self.ghost:
            for r in self.boxes:
                if r["frame"] == self.f - 1:
                    self._poly(can, r, (150, 150, 150), 1, dashed=False)
        for r in self.boxes:
            if r["frame"] == self.f:
                col = self.COLS[r["track_id"] % len(self.COLS)]
                self._poly(can, r, col, 2 if r["track_id"] == self.track else 1, dashed=not r["fitted"])
                sx, sy = self.to_scr(r["x_max"], r["y_min"])
                cv2.putText(can, str(r["track_id"]), (int(sx) + 3, int(sy)), 0, 0.45, col, 1, cv2.LINE_AA)
        if self.drag:
            (ax, ay), (bx, by) = self.drag
            cv2.rectangle(can, (int(ax), int(ay)), (int(bx), int(by)), (255, 255, 255), 1)
        n_here = sum(r["frame"] == self.f for r in self.boxes)
        tags = (f"frame {self.f}/{self.frames.n - 1}"
                f"{'  held-out' if is_heldout(self.name, self.f) else ''}   ember {self.track}   "
                f"boxes here {n_here}, total {len(self.boxes)}   view {self.VIEWS[self.view]}   "
                f"zoom {self.zoom:.2f}x   fit {'on' if self.fitting else 'OFF'}   {self.msg}")
        cv2.rectangle(can, (0, 0), (self.vw, 24), (0, 0, 0), -1)
        cv2.putText(can, tags, (8, 17), 0, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        if self.help:
            lines = ["drag: box an ember (fitted tight)   click box: continue its ember   right-click: delete",
                     "n new ember   u undo   r fitting on/off   d/a frame +-1   ] [ +-10   h next held-out",
                     "+ - wheel zoom   i j k l pan   0 whole frame   m view   g ghosts   ? help   q quit (saves)"]
            cv2.rectangle(can, (0, self.vh - 84), (self.vw, self.vh), (0, 0, 0), -1)
            for k, t in enumerate(lines):
                cv2.putText(can, t, (8, self.vh - 62 + 20 * k), 0, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return can

    def _poly(self, can, r, col, th, dashed):
        p = np.array([self.to_scr(x, y) for x, y in corners(r)], np.float32)
        if dashed:
            for a, b in zip(p, np.roll(p, -1, 0)):
                for t in np.linspace(0, 1, 9)[::2]:
                    q0, q1 = a + (b - a) * t, a + (b - a) * min(t + 0.125, 1)
                    cv2.line(can, tuple(q0.astype(int)), tuple(q1.astype(int)), col, th, cv2.LINE_AA)
        else:
            cv2.polylines(can, [np.round(p).astype(np.int32)], True, col, th, cv2.LINE_AA)

    # -- editing -----------------------------------------------------------------
    def box_at(self, ix, iy):
        for r in sorted((r for r in self.boxes if r["frame"] == self.f), key=lambda r: r["L"] * r["W"]):
            if cv2.pointPolygonTest(corners(r).reshape(-1, 1, 2), (ix, iy), True) >= -2 / max(self.zoom, 1e-6):
                return r
        return None

    def add(self, x0, y0, x1, y1):
        f = self.frames.get(self.f)
        fit = fit_box(f, x0, y0, x1, y1) if self.fitting else None
        r = dict(fit) if fit else raw_box(x0, y0, x1, y1)
        r.update(frame=self.f, track_id=self.track, fitted=int(fit is not None))
        with_bounds(r)
        # one box per ember per frame: a second box for the same ember replaces the first
        self.boxes = [b for b in self.boxes if not (b["frame"] == self.f and b["track_id"] == self.track)]
        self.boxes.append(r)
        self.history.append(r)
        self.msg = "fitted" if fit else ("too faint to fit: kept as drawn" if self.fitting else "kept as drawn")
        save(self.out, self.boxes)

    def undo(self):
        while self.history:
            r = self.history.pop()
            if r in self.boxes:
                self.boxes.remove(r)
                save(self.out, self.boxes)
                self.msg = f"undone: ember {r['track_id']} on frame {r['frame']}"
                return

    def mouse(self, ev, sx, sy, flags, _param):
        if ev == cv2.EVENT_LBUTTONDOWN:
            self.drag = [(sx, sy), (sx, sy)]
        elif ev == cv2.EVENT_MOUSEMOVE and self.drag:
            self.drag[1] = (sx, sy)
        elif ev == cv2.EVENT_LBUTTONUP and self.drag:
            # The end point is where the button came up, not the last move event:
            # a quick drag may send no move at all, and would read as a click.
            (ax, ay), (bx, by) = self.drag[0], (sx, sy)
            self.drag = None
            if abs(bx - ax) < 4 and abs(by - ay) < 4:            # a click: pick an ember
                r = self.box_at(*self.to_img(sx, sy))
                if r is not None:
                    self.track = r["track_id"]
                    self.msg = f"continuing ember {self.track}"
                return
            (x0, y0), (x1, y1) = self.to_img(ax, ay), self.to_img(bx, by)
            self.add(x0, y0, x1, y1)
        elif ev == cv2.EVENT_RBUTTONDOWN:
            r = self.box_at(*self.to_img(sx, sy))
            if r is not None:
                self.boxes.remove(r)
                save(self.out, self.boxes)
                self.msg = f"deleted ember {r['track_id']} on frame {r['frame']}"
        elif ev == cv2.EVENT_MOUSEWHEEL:
            d = wheel_delta(flags)
            if d:
                self.zoom_at(sx, sy, 1.25 if d > 0 else 0.8)

    def key(self, k):
        c = chr(k) if 0 <= k < 256 else ""
        step = 120 / self.zoom
        if c in ("q", "\x1b"):
            return False
        if c in ("d", "."):
            self.go(self.f + 1, +1)
        elif c in ("a", ","):
            self.go(self.f - 1, -1)
        elif c == "]":
            self.go(self.f + 10, +1)
        elif c == "[":
            self.go(self.f - 10, -1)
        elif c == "h":
            self.next_heldout()
        elif c in ("+", "="):
            self.zoom_at(self.vw / 2, self.vh / 2, 1.5)
        elif c == "-":
            self.zoom_at(self.vw / 2, self.vh / 2, 1 / 1.5)
        elif c == "0":
            self.fit_whole()
        elif c == "i":
            self.cy -= step
        elif c == "k":
            self.cy += step
        elif c == "j":
            self.cx -= step
        elif c == "l":
            self.cx += step
        elif c == "n":
            self.track = max([self.track] + [r["track_id"] for r in self.boxes]) + 1
            self.msg = f"new ember {self.track}"
        elif c == "u":
            self.undo()
        elif c == "r":
            self.fitting = not self.fitting
        elif c == "m":
            self.view = (self.view + 1) % len(self.VIEWS)
        elif c == "g":
            self.ghost = not self.ghost
        elif c == "?":
            self.help = not self.help
        return True

    def run(self):
        win = "firebrand tagger"
        cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(win, self.mouse)
        while True:
            cv2.imshow(win, self.render())
            k = cv2.waitKey(20)
            if k != -1 and not self.key(k & 0xFF):
                break
        save(self.out, self.boxes)
        cv2.destroyAllWindows()
        print(f"saved {len(self.boxes)} boxes, {len({r['track_id'] for r in self.boxes})} embers -> {self.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("video")
    ap.add_argument("--out", help="CSV to write (default: <video>_manual.csv beside the video)")
    ap.add_argument("--heldout", action="store_true", help="only the kit's held-out frames")
    ap.add_argument("--start", type=int, default=0, help="first frame to show")
    ap.add_argument("--view", default="1600x900", help="window size, e.g. 1280x720")
    a = ap.parse_args()
    out = a.out or str(Path(a.video).with_name(Path(a.video).stem + "_manual.csv"))
    vw, vh = (int(v) for v in a.view.lower().split("x"))
    Tagger(a.video, out, heldout=a.heldout, start=a.start, view=(vw, vh)).run()


if __name__ == "__main__":
    main()
