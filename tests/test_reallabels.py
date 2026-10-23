#!/usr/bin/env python3
"""Checks for training on real firebrands (firebrand/reallabels.py).

    python tests/test_reallabels.py

No video, no GPU, a second. Run it after touching reallabels.py.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from firebrand import reallabels as RL  # noqa: E402
from firebrand.negatives import is_heldout  # noqa: E402


def streak(img, x, y, L, th, val, w=3):
    dx, dy = np.cos(th) * L / 2, np.sin(th) * L / 2
    cv2.line(img, (int(x - dx), int(y - dy)), (int(x + dx), int(y + dy)), int(val), w, cv2.LINE_AA)


def tile():
    rng = np.random.default_rng(0)
    cen = ((rng.random((256, 256)) < 0.02) * rng.integers(1, 4, (256, 256))).astype(np.uint8)
    streak(cen, 60, 60, 20, 0.3, 120)          # labelled
    streak(cen, 180, 190, 30, -1.0, 90)        # labelled
    streak(cen, 200, 50, 16, 1.2, 70)          # NOT labelled
    rows = np.array([[60, 60, 22, 3, 0.3], [180, 190, 32, 3, -1.0]], np.float32)
    return cen, rows


def test_unlabelled_removed_labelled_kept():
    cen, rows = tile()
    before = cen.copy()
    RL.clean_centre(cen, rows, 0, 0, thr=4, peak_thr=6)
    assert (cen[50:71, 45:76] == before[50:71, 45:76]).all(), "labelled ember changed"
    assert cen[175:206, 162:199].max() > 50, "labelled ember removed"
    assert cen[38:63, 188:213].max() < 4, "unlabelled ember still there"
    faint = (before > 0) & (before < 4)
    assert (cen[faint] == before[faint]).mean() > 0.9, "camera grain was scrubbed"


def test_merged_glow_does_not_save_an_unlabelled_ember():
    """A bright unlabelled ember whose glow touches a labelled one must still go."""
    cen = np.zeros((256, 256), np.uint8)
    streak(cen, 100, 100, 14, 0.0, 60)                     # labelled, small
    cv2.circle(cen, (124, 100), 9, 200, -1)                # bright, unlabelled, touching
    rows = np.array([[100, 100, 15, 3, 0.0]], np.float32)
    RL.clean_centre(cen, rows, 0, 0, thr=4, peak_thr=6)
    assert cen[92:109, 118:134].max() < 4, "unlabelled ember survived via the merged glow"
    assert cen[98:103, 95:105].max() > 30, "labelled ember lost"


def test_targets_on_labels_only():
    _, rows = tile()
    y = RL.draw_targets(rows, 0, 0, 256)
    assert y[55:66, 50:71].max() > 0.5 and y[175:206, 165:196].max() > 0.5
    assert y[40:61, 190:211].max() == 0


def test_heldout_blocks_never_trained():
    labels = {f: np.zeros((1, 5), np.float32) for f in range(0, 640)}
    runs = RL.label_runs("clipX", "/x/clipX.mp4", 24.0, labels, 640)
    assert runs and not any(is_heldout("clipX", f) for r in runs for f in range(r.a, r.b + 1))
    assert len(runs) == 15                                  # 20 blocks, 5 held out


def test_stale_review_ignored():
    with tempfile.TemporaryDirectory() as d:
        t = Path(d) / "tags" / "c1"
        t.mkdir(parents=True)
        (t / "reject.txt").write_text("3\n")
        old = time.time() - 100
        os.utime(t / "reject.txt", (old, old))
        (t / "done.json").write_text(json.dumps({"n_frames": 10}))
        assert not RL.review_is_current(d, "c1"), "a review older than the tags was used"
        (t / "reject.txt").write_text("3\n")
        assert RL.review_is_current(d, "c1")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
