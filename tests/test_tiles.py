#!/usr/bin/env python3
"""Checks for how tiles are held and saved (firebrand/dataset.py). No GPU, seconds.

    python tests/test_tiles.py

The tiles stage holds gigabytes on a 15.6 GB VM, so masks stay uint8 throughout.
These pin down that doing so changes nothing that reaches the model.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from firebrand import dataset as DS  # noqa: E402
from firebrand.config import Config  # noqa: E402


def masks(n=7, t=32):
    rng = np.random.default_rng(0)
    y = np.zeros((n, t, t), np.float32)
    y[:, 10:12, 5:25] = rng.random((n, 2, 20)).astype(np.float32)
    y[:, 20, 5:25] = 1.0
    return y


def test_mask8_matches_what_saving_always_did():
    y = masks()
    assert np.array_equal(DS.to_mask8(y, block=3), (y * 255).astype(np.uint8))


def test_mask8_is_not_a_truncating_cast():
    """Regression trap: 0..1 float masks through astype(uint8) become all zeros."""
    y = masks()
    assert y.astype(np.uint8)[:, 10:12].max() == 0          # what the trap does
    assert DS.to_mask8(y)[:, 20, 5:25].min() == 255


def test_mask8_leaves_uint8_alone():
    y = DS.to_mask8(masks())
    assert DS.to_mask8(y) is y


def test_concat_free_same_as_concatenate():
    a = [np.arange(12, dtype=np.uint8).reshape(3, 4) + i for i in range(4)]
    want = np.concatenate(a)
    parts = list(a)
    got = DS.concat_free(parts)
    assert np.array_equal(got, want) and got.dtype == want.dtype
    assert parts == [], "parts must be released as they are copied"


def test_saved_file_is_identical_either_way():
    rng = np.random.default_rng(1)
    x = rng.integers(0, 255, (7, 3, 32, 32), dtype=np.uint8)
    y = masks()
    with tempfile.TemporaryDirectory() as d:
        DS.save_tiles(Path(d) / "f.npz", x, y)
        DS.save_tiles(Path(d) / "u.npz", x, DS.to_mask8(y))
        a, b = DS.load_tiles(Path(d) / "f.npz", as_float=False), \
            DS.load_tiles(Path(d) / "u.npz", as_float=False)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    assert a[1].dtype == np.uint8


def test_stats_and_pos_weight_agree():
    x = np.zeros((7, 3, 32, 32), np.uint8)
    y = masks()
    s8, sf = DS.tile_stats(x, DS.to_mask8(y)), DS.tile_stats(x, y)
    assert abs(s8["positive_pixel_frac"] - sf["positive_pixel_frac"]) < 1e-3
    assert abs(DS.suggested_pos_weight(DS.to_mask8(y)) - DS.suggested_pos_weight(y)) < 1.0


def test_build_from_synthetic_u8():
    cfg = Config()
    cfg.synth.reencode_h264 = False                # no ffmpeg needed
    cfg.detect.tophat_ksize = 15
    cfg.detect.stabilize = False
    cfg.train.tile, cfg.train.stride = 128, 128
    bg = [np.full((256, 256, 3), 40, np.uint8) for _ in range(4)]
    a = DS.build_from_synthetic(bg, 20, cfg, seed=1, progress=False)
    b = DS.build_from_synthetic(bg, 20, cfg, seed=1, progress=False, masks_u8=True)
    assert len(a[0]) and b[1].dtype == np.uint8 and a[1].dtype == np.float32
    assert np.array_equal(a[0], b[0])
    assert np.array_equal(DS.to_mask8(a[1]), b[1])


if __name__ == "__main__":
    for k, v in list(globals().items()):
        if k.startswith("test_"):
            v()
            print("ok ", k)
