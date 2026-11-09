#!/usr/bin/env python3
"""Checks for the manual tagging tool (tag_tool.py), without a screen. Seconds.

    python tests/test_tag_tool.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import tag_tool as TT  # noqa: E402


def scene_with_streak(cx=200, cy=120, L=30, angle_deg=30, peak=180, bg=40):
    img = np.full((240, 400, 3), (bg, 90, 160), np.uint8)       # orange-ish ground, blue = bg
    th = np.deg2rad(angle_deg)
    dx, dy = np.cos(th) * L / 2, np.sin(th) * L / 2
    layer = np.zeros(img.shape[:2], np.float32)
    cv2.line(layer, (int(cx - dx), int(cy - dy)), (int(cx + dx), int(cy + dy)), 1.0, 2, cv2.LINE_AA)
    layer = cv2.GaussianBlur(layer, (0, 0), 0.8)
    layer /= layer.max()
    img[..., 0] = np.clip(bg + layer * (peak - bg), 0, 255).astype(np.uint8)
    return img


def test_fit_is_tight_and_follows_the_angle():
    img = scene_with_streak()
    r = TT.fit_box(img, 170, 95, 235, 150)                      # a loose box round it
    assert r is not None
    assert abs(r["x"] - 200) < 1.5 and abs(r["y"] - 120) < 1.5, r
    ang = np.rad2deg(r["theta"]) % 180
    assert min(abs(ang - 30), 180 - abs(ang - 30)) < 4, ang
    assert 26 <= r["L"] <= 36 and r["W"] <= 5, r                # tight: not the 65 x 55 drawn


def test_too_faint_is_none():
    img = scene_with_streak(peak=44)                            # 4 above background
    assert TT.fit_box(img, 170, 95, 235, 150) is None
    assert TT.fit_box(img, 10, 10, 10, 10) is None              # a zero-size box


def test_save_load_round_trip():
    r = TT.with_bounds(dict(TT.fit_box(scene_with_streak(), 170, 95, 235, 150),
                            frame=7, track_id=3, fitted=1))
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "m.csv"
        TT.save(p, [r])
        back = TT.load(p)
    assert len(back) == 1 and back[0]["frame"] == 7 and back[0]["track_id"] == 3
    for k in ("x", "y", "L", "W", "theta"):
        assert abs(back[0][k] - r[k]) < 1e-3, k


def test_heldout_matches_the_kit():
    from firebrand.negatives import is_heldout
    for name in ("clip5", "clip9", "D01_20250107230423"):
        for f in range(0, 400, 7):
            assert TT.is_heldout(name, f) == is_heldout(name, f), (name, f)


def test_wheel_delta():
    # the step sits in the signed high 16 bits of flags; low bits are buttons/keys
    assert TT.wheel_delta(120 << 16) == 120
    assert TT.wheel_delta((120 << 16) | cv2.EVENT_FLAG_CTRLKEY) == 120
    assert TT.wheel_delta(-120 << 16) == -120                  # a negative Python int
    assert TT.wheel_delta((-120 << 16) & 0xFFFFFFFF) == -120   # or the same as unsigned 32-bit
    assert TT.wheel_delta(0) == 0


def _session(tmp, **kw):
    clip = ROOT / "work" / "recall_review" / "clips" / "clip9.mp4"
    if not clip.exists():
        return None
    return TT.Tagger(clip, Path(tmp) / "clip9_manual.csv", view=(1280, 720), **kw)


def test_session_draw_continue_delete_undo():
    with tempfile.TemporaryDirectory() as d:
        t = _session(d, start=160)
        if t is None:
            print("   (clip9 not here: session test skipped)")
            return
        assert t.render().shape == (720, 1280, 3)
        # zoom on a real tagged ember and box it
        tags = ROOT / "work" / "tags" / "clip9" / "detections.csv"
        import pandas as pd
        e = pd.read_csv(tags)
        e = e[e.frame == 160].copy()
        # a clearly visible ember: brightest in the image itself (the model's own
        # score saturates at 254 and says nothing about how visible an ember is)
        img = t.frames.get(160)
        e["blue"] = [int(img[int(r.y), int(r.x), 0]) for r in e.itertuples()]
        e = e.sort_values("blue", ascending=False).iloc[0]
        m = e.L / 2 + 6                                         # a box drawn round it, with room
        t.zoom = 6.0
        t.cx, t.cy = e.x, e.y
        sx, sy = t.to_scr(e.x - m, e.y - m)
        ex, ey = t.to_scr(e.x + m, e.y + m)
        t.mouse(cv2.EVENT_LBUTTONDOWN, int(sx), int(sy), 0, None)
        t.mouse(cv2.EVENT_MOUSEMOVE, int(ex), int(ey), 0, None)
        t.mouse(cv2.EVENT_LBUTTONUP, int(ex), int(ey), 0, None)
        assert len(t.boxes) == 1
        b = t.boxes[0]
        print(f"   real ember: model x,y,L,W = {e.x:.1f},{e.y:.1f},{e.L:.1f},{e.W:.1f}; "
              f"fitted = {b['x']:.1f},{b['y']:.1f},{b['L']:.1f},{b['W']:.1f} (fitted={b['fitted']})")
        assert b["fitted"] == 1 and np.hypot(b["x"] - e.x, b["y"] - e.y) < 6
        assert TT.load(t.out)[0]["frame"] == 160                # saved at once
        # next frame, new ember, then click the old box on frame 160 to continue it
        t.key(ord("n")); first = b["track_id"]
        assert t.track == first + 1
        t.key(ord("d")); assert t.f == 161
        t.key(ord("a")); assert t.f == 160
        cx, cy = t.to_scr(b["x"], b["y"])
        t.mouse(cv2.EVENT_LBUTTONDOWN, int(cx), int(cy), 0, None)
        t.mouse(cv2.EVENT_LBUTTONUP, int(cx), int(cy), 0, None)
        assert t.track == first
        t.mouse(cv2.EVENT_RBUTTONDOWN, int(cx), int(cy), 0, None)     # delete it
        assert t.boxes == [] and TT.load(t.out) == []
        # a quick drag with NO move event between press and release still draws a box
        # (regression: the end point used to come from the last move event)
        t.mouse(cv2.EVENT_LBUTTONDOWN, int(sx), int(sy), 0, None)
        t.mouse(cv2.EVENT_LBUTTONUP, int(ex), int(ey), 0, None)
        assert len(t.boxes) == 1 and len(TT.load(t.out)) == 1
        t.key(ord("u"))
        assert t.boxes == [] and TT.load(t.out) == []
        for c in "m" * 3 + "g?r0+-ijkl":
            assert t.key(ord(c))
            t.render()
        assert t.key(ord("q")) is False


def test_heldout_only_navigation():
    with tempfile.TemporaryDirectory() as d:
        t = _session(d, heldout=True)
        if t is None:
            return
        seen = {t.f}
        for _ in range(80):
            t.key(ord("d"))
            seen.add(t.f)
        assert all(TT.is_heldout("clip9", f) for f in seen), sorted(seen)[:10]
        assert len(seen) > 40


if __name__ == "__main__":
    for k, v in list(globals().items()):
        if k.startswith("test_"):
            v()
            print("ok ", k)
