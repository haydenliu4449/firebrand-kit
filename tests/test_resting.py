#!/usr/bin/env python3
"""Checks for the resting-ember pass (firebrand/resting.py). No video, seconds.

    python tests/test_resting.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from firebrand import resting as R  # noqa: E402


def scene_frame(rng, h=240, w=320):
    """Dark ground with some dim static texture, as BGR."""
    b = np.full((h, w), 25, np.uint8)
    b[60:64, :] = 70                       # a curb line, in the scene
    noise = rng.integers(0, 3, (h, w)).astype(np.uint8)
    return cv2.merge([b + noise, b + noise, b + noise])


def dot(f, x, y, v=220, r=2):
    cv2.circle(f, (x, y), r, (v, v, v), -1)


def clip(n=80):
    rng = np.random.default_rng(0)
    out = []
    for i in range(n):
        f = scene_frame(rng)
        if 20 <= i < 70:
            dot(f, 100, 150)               # lands at 20, gone at 70
        dot(f, 200, 100, v=90)             # dim: texture, not an ember
        dot(f, 10 + 3 * i, 200)            # flying: never in a median
        out.append(f)
    return out


def test_finds_only_the_resting_ember():
    rng = np.random.default_rng(1)
    empty = scene_frame(rng)
    cfg = R.RestConfig()
    scene = R.tophat(empty[..., 0], 31)
    rows = R.run_clip(iter(clip()), scene, cfg=cfg)
    assert len(rows) == 1, rows
    e = rows[0]
    assert abs(e["x"] - 100) < 1.5 and abs(e["y"] - 150) < 1.5, e
    assert 20 <= e["frame_start"] <= 40 and 50 <= e["frame_end"] <= 72, e


def test_scene_is_not_an_ember():
    """A bright spot that is in the scene reference is the scene."""
    rng = np.random.default_rng(2)
    f = scene_frame(rng)
    dot(f, 50, 50)
    assert R.find(f[..., 0], R.tophat(f[..., 0], 31), R.RestConfig(), 0) == []


def test_no_ember_beside_strong_scene():
    """A dot right next to a wall edge cannot be told from the edge."""
    rng = np.random.default_rng(3)
    empty = scene_frame(rng)
    scene = R.tophat(empty[..., 0], 31)     # curb at y 60-63 is strong here
    f = empty.copy()
    dot(f, 120, 68)                         # beside the curb
    dot(f, 120, 150)                        # on plain ground
    got = R.find(f[..., 0], scene, R.RestConfig(), 0)
    assert len(got) == 1 and abs(got[0][2] - 150) < 1.5, got


def test_group_needs_two_looks():
    cfg = R.RestConfig()
    rows = R.group([(0, 10.0, 10.0, 90.0, 9), (12, 11.0, 10.0, 90.0, 9),
                    (12, 200.0, 10.0, 90.0, 9)], cfg)
    assert len(rows) == 1 and rows[0]["looks"] == 2, rows


if __name__ == "__main__":
    for k, v in list(globals().items()):
        if k.startswith("test_"):
            v()
            print("ok ", k)
