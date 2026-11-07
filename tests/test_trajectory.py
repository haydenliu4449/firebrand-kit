#!/usr/bin/env python3
"""Checks for joining hops into trajectories (firebrand/trajectory.py). Seconds.

    python tests/test_trajectory.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from firebrand import track as T, trajectory as TJ  # noqa: E402
from firebrand.detect import Detection  # noqa: E402


def det(f, x, y):
    return Detection(f, float(x), float(y), 10.0, 2.0, 0.0, 100.0, 20, 500.0)


def track(i, pts, f0):
    return T.Track(i, [det(f0 + k, x, y) for k, (x, y) in enumerate(pts)])


FALL = [(100, 100), (110, 120), (120, 140), (130, 160)]       # down and right
RISE = [(138, 152), (146, 140), (154, 130)]                    # up and right, slower


def test_a_bounce_is_one_trajectory():
    rows, tof = TJ.build([track(0, FALL, 0), track(1, RISE, 4)])
    assert len(rows) == 1 and rows[0]["n_hops"] == 1, rows
    assert tof == {0: 0, 1: 0}
    assert rows[0]["contacts"].startswith("3:130.0,160.0")


def test_not_a_bounce_stays_apart():
    cases = [
        [track(0, FALL, 0), track(1, [(140, 170), (150, 190)], 4)],       # still falling
        [track(0, FALL, 0), track(1, [(122, 152), (114, 140)], 4)],       # turns back left
        [track(0, FALL, 0), track(1, [(138, 152), (166, 110)], 4)],       # rises faster
        [track(0, FALL, 0), track(1, [(400, 160), (408, 148)], 4)],       # far away
        [track(0, FALL, 0), track(1, RISE, 9)],                           # long after
    ]
    for ts in cases:
        rows, _ = TJ.build(ts)
        assert len(rows) == 2 and all(r["n_hops"] == 0 for r in rows), ts


def test_each_end_used_once():
    a, b = track(0, FALL, 0), track(1, RISE, 4)
    c = track(2, [(139, 151), (147, 139), (155, 129)], 4)                 # a second candidate
    rows, tof = TJ.build([a, b, c])
    assert sum(r["n_hops"] for r in rows) == 1 and len(rows) == 2


def test_landing_on_a_resting_ember():
    rest = [{"ember_id": 7, "x": 131.0, "y": 161.0, "frame_start": 12},
            {"ember_id": 8, "x": 900.0, "y": 900.0, "frame_start": 5}]
    rows, _ = TJ.build([track(0, FALL, 0)], rest)
    assert rows[0]["landed_resting_id"] == 7
    rows, _ = TJ.build([track(0, FALL, 0)], [dict(rest[0], frame_start=200)])
    assert rows[0]["landed_resting_id"] == -1                             # too late


if __name__ == "__main__":
    for k, v in list(globals().items()):
        if k.startswith("test_"):
            v()
            print("ok ", k)
