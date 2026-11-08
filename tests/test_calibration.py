#!/usr/bin/env python3
"""Checks for pixels -> metres (firebrand/calibration.py, calib/left_front.json). Seconds.

    python tests/test_calibration.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from firebrand.calibration import GroundCalibration  # noqa: E402
from firebrand import trajectory as TJ  # noqa: E402

CAL = GroundCalibration.load(ROOT / "calib" / "left_front.json")


def test_round_trip():
    p = np.array([[1000, 1500], [3000, 800], [2500, 2000], [3500, 400], [600, 1900]], float)
    assert np.abs(CAL.to_pixel(CAL.to_ground(p)) - p).max() < 1e-6


def test_undistort_inverts():
    p = np.array([[100, 100], [3800, 2100], [1920, 1080]], float)
    assert np.abs(CAL.distort(CAL.undistort(p)) - p).max() < 1e-6


def test_centre_unmoved_and_edges_pushed_out():
    c = np.array([[1920.0, 1080.0]])
    assert np.abs(CAL.undistort(c) - c).max() < 1e-9
    corner = np.array([[10.0, 10.0]])
    assert np.linalg.norm(CAL.undistort(corner) - c) > np.linalg.norm(corner - c)   # barrel undone


def test_slab_lattice_on_the_ground():
    """A slab's corners, mapped to the image and back, stay 1.2 x 0.6 m apart, and the
    whole driveway sits in front of the camera (finite, positive scale)."""
    X = np.array([[0, 0], [1.2, 0], [0, 0.6], [1.2, 0.6]], float)
    px = CAL.to_pixel(X)
    assert np.isfinite(px).all()
    back = CAL.to_ground(px)
    assert np.allclose(np.linalg.norm(back[1] - back[0]), 1.2, atol=1e-6)
    assert np.allclose(np.linalg.norm(back[2] - back[0]), 0.6, atol=1e-6)


def test_ground_frame_is_a_map_not_a_mirror():
    """Image y points down, map y points up: a map that is not mirrored has a
    NEGATIVE Jacobian determinant everywhere. The first fit was mirrored (the
    'Left Front' label read backwards on a top-down render)."""
    for p in ([1920, 1600], [1000, 1800], [3000, 1300], [2500, 800]):
        g0 = CAL.to_ground([p])[0]
        gx = CAL.to_ground([[p[0] + 1.0, p[1]]])[0] - g0
        gy = CAL.to_ground([[p[0], p[1] + 1.0]])[0] - g0
        assert np.linalg.det(np.c_[gx, gy]) < 0, p


def test_hops_in_metres():
    rows = [dict(trajectory_id=0, contacts="10:2000.0,1500.0;20:2100.0,1500.0", landed_resting_id=-1),
            dict(trajectory_id=1, contacts="", landed_resting_id=-1)]
    TJ.add_metres(rows, [], CAL)
    g = CAL.to_ground(np.array([[2000.0, 1500.0], [2100.0, 1500.0]]))
    assert abs(float(rows[0]["hops_m"]) - np.linalg.norm(g[1] - g[0])) < 1e-3
    assert rows[1]["hops_m"] == ""
    assert CAL.provisional                     # the slab size is still assumed


if __name__ == "__main__":
    for k, v in list(globals().items()):
        if k.startswith("test_"):
            v()
            print("ok ", k)
