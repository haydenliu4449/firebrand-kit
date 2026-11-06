#!/usr/bin/env python3
"""Checks for core splitting (detect.split_cores). No video, under a second.

    python tests/test_detect.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from firebrand import detect as D  # noqa: E402
from firebrand.config import DetectConfig  # noqa: E402


def blob(res, thr=10):
    ys, xs = np.where(res > thr)
    return ys, xs


def test_two_embers_in_one_glow_come_apart():
    res = np.zeros((40, 80), np.uint8)
    res[15:25, 5:75] = 20                  # shared glow
    res[19:21, 10:30] = 200                # bright streak
    res[19:21, 50:70] = 180                # and another
    parts = D.split_cores(*blob(res), res, 0.5, 5)
    assert len(parts) >= 2, len(parts)
    cx = sorted(float(p[1].mean()) for p in parts[:2])
    assert abs(cx[0] - 19.5) < 2 and abs(cx[1] - 59.5) < 2, cx


def test_faint_streak_is_left_alone():
    res = np.zeros((20, 60), np.uint8)
    res[9:11, 10:50] = 30
    ys, xs = blob(res)
    assert [len(p[0]) for p in D.split_cores(ys, xs, res, 0.5, 5)] == [len(ys)]


def test_core_filling_its_box():
    """Regression: the bright core spans the whole box of what is left, so the
    box has no zero pixel, the distance transform came back huge, and cv2.dilate
    crashed on the size (seen on clip9: it would have killed a tag job)."""
    res = np.full((3, 5), 200, np.uint8)
    res[0, 0] = 10                         # one dim pixel keeps peak >> lower quartile...
    res[0, 1] = res[0, 2] = res[0, 3] = res[0, 4] = 10
    ys, xs = np.where(res > 0)
    parts = D.split_cores(ys[5:], xs[5:], res, 0.5, 5)   # all core
    assert len(parts) == 1
    res2 = np.full((4, 6), 200, np.uint8)
    res2[0, :] = 0
    res2[1, 0] = 40
    res2[1, 1] = 40
    res2[1, 2] = 40
    ys, xs = np.where(res2 > 0)
    assert D.split_cores(ys, xs, res2, 0.5, 5)
    full = np.full((5, 5), 100, np.uint8)
    full[0, 0] = full[0, 1] = full[0, 2] = full[1, 0] = full[1, 1] = full[2, 0] = 10
    full[0, 3] = full[0, 4] = full[1, 2] = 10
    ys, xs = np.where(full > 0)            # lower quartile 10, peak 100; core is no box
    assert D.split_cores(ys, xs, full, 0.05, 5)   # frac so low the core IS the whole box


def test_off_means_off():
    res = np.zeros((20, 60), np.uint8)
    res[8:12, 5:55] = 15
    res[9:11, 10:20] = 200
    ys, xs = blob(res)
    assert len(D.split_cores(ys, xs, res, 0.0, 5)) == 1


def test_detect_frame_runs():
    res = np.zeros((60, 120), np.uint8)
    res[20:22, 10:40] = 200
    assert len(D.detect_frame(res, 0, DetectConfig())) >= 0


if __name__ == "__main__":
    for k, v in list(globals().items()):
        if k.startswith("test_"):
            v()
            print("ok ", k)
