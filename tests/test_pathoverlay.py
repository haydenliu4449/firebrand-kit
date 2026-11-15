#!/usr/bin/env python3
"""Checks for the whole-path overlay (firebrand/pathoverlay.py, path_overlay.py,
run_all.write_tag_overlay). No footage needed, seconds.

    python tests/test_pathoverlay.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from firebrand import pathoverlay as PO, track as T, trajectory as TJ  # noqa: E402
from firebrand.detect import Detection  # noqa: E402

SHAPE = (240, 400, 3)
GREY = 40                                       # the empty scene


def det(f, x, y, L=10.0, W=2.0, theta=0.0):
    return Detection(f, float(x), float(y), L, W, theta, 100.0, 20, 500.0)


def track(i, pts, f0, step=1):
    return T.Track(i, [det(f0 + k * step, x, y) for k, (x, y) in enumerate(pts)])


def blank():
    return np.full(SHAPE, GREY, np.uint8)


def painted(img, x, y, r=1):
    """Is anything drawn within r px of (x, y)?"""
    x, y = int(round(x)), int(round(y))
    return bool((img[y - r:y + r + 1, x - r:x + r + 1] != GREY).any())


def is_colour(img, x, y, col, tol=40):
    px = img[int(round(y)), int(round(x))].astype(int)
    return bool(np.abs(px - np.array(col)).max() <= tol)


A = [(50, 60), (90, 60), (130, 60), (170, 60)]          # flies right, frames 0-3


def test_earlier_positions_stay_and_are_joined():
    """The request itself: seen on frame 0, so on frame 1 its frame-0 position is
    still drawn and joined towards where it is now; by the end, one unbroken line."""
    o = PO.PathOverlay(PO.from_tracks([track(7, A, 0)]), SHAPE, labels=False)
    f0 = o.draw(blank(), 0)
    assert not (f0 != GREY).any()                       # first sighting: nothing yet
    f1 = o.draw(blank(), 1)
    assert painted(f1, 50, 60)                          # where it was on frame 0
    assert painted(f1, 70, 60)                          # the line towards frame 1's position
    f9 = o.draw(blank(), 9)                             # long after it was last seen
    for x in range(50, 171, 5):                         # one continuous streak, start to end
        assert painted(f9, x, 60), x
    assert is_colour(f9, 110, 60, PO.colour(7))
    assert not painted(f9, 200, 60) and not painted(f9, 110, 100)


def test_the_ember_in_view_is_never_drawn_over():
    o = PO.PathOverlay(PO.from_tracks([track(0, A, 0)]), SHAPE, labels=False)
    f2 = o.draw(blank(), 2)                             # detected at (130, 60), L 10
    assert painted(f2, 110, 60)                         # the line comes towards it...
    box = f2[60 - 4:60 + 5, 130 - 9:130 + 10]           # ...and stops short of its box (pad 4)
    assert not (box != GREY).any()
    f3 = o.draw(blank(), 3)
    assert painted(f3, 130, 60, r=0)                    # a frame later that position is on the path


def test_each_ember_keeps_its_own_colour():
    a, b = track(1, A, 0), track(2, [(x, y + 80) for x, y in A], 0)
    f = PO.PathOverlay(PO.from_tracks([a, b]), SHAPE, labels=False).draw(blank(), 9)
    assert PO.colour(1) != PO.colour(2)
    assert is_colour(f, 110, 60, PO.colour(1)) and is_colour(f, 110, 140, PO.colour(2))
    assert PO.colour(1) == PO.colour(1 + len(PO.PALETTE))       # colours repeat, IDs do not


def test_a_trajectory_is_one_path_across_its_bounce():
    fall = track(0, [(100, 100), (110, 120), (120, 140), (130, 160)], 0)
    rise = track(1, [(138, 152), (146, 140), (154, 130)], 5)    # a frame missing at the bounce
    rows, traj_of = TJ.build([fall, rise])
    assert traj_of == {0: 0, 1: 0}
    by_track = PO.from_tracks([fall, rise])
    by_ember = PO.from_tracks([fall, rise], traj_of)
    assert len(by_track) == 2 and len(by_ember) == 1 and len(by_ember[0].frames) == 7
    mid = (134, 156)                                    # between the contact and the first rise point
    assert not painted(PO.PathOverlay(by_track, SHAPE, labels=False).draw(blank(), 20), *mid)
    assert painted(PO.PathOverlay(by_ember, SHAPE, labels=False).draw(blank(), 20), *mid)


def test_linger_takes_a_path_away_and_default_keeps_it():
    paths = PO.from_tracks([track(0, A, 0)])
    o = PO.PathOverlay(paths, SHAPE, linger=2, labels=False)
    assert painted(o.draw(blank(), 5), 110, 60)         # last seen on 3: still there on 5
    assert not (o.draw(blank(), 6) != GREY).any()       # gone on 6
    keep = PO.PathOverlay(paths, SHAPE, labels=False)
    assert painted(keep.draw(blank(), 500), 110, 60)


def test_linger_matches_the_kept_layer_while_everything_is_in_view():
    ts = [track(0, A, 0), track(1, [(60, 200), (100, 170), (140, 150)], 1)]
    a = PO.PathOverlay(PO.from_tracks(ts), SHAPE, labels=False)
    b = PO.PathOverlay(PO.from_tracks(ts), SHAPE, linger=100, labels=False)
    for i in range(6):
        assert np.array_equal(a.draw(blank(), i), b.draw(blank(), i)), i


def test_starting_part_way_shows_the_path_so_far():
    o = PO.PathOverlay(PO.from_tracks([track(0, A, 0)]), SHAPE, labels=False)
    f = o.draw(blank(), 3)                              # first call is frame 3
    assert painted(f, 70, 60) and painted(f, 110, 60)


def test_missing_frames_are_bridged():
    o = PO.PathOverlay(PO.from_tracks([track(0, A, 0, step=3)]), SHAPE, labels=False)   # 0, 3, 6, 9
    f = o.draw(blank(), 4)
    assert painted(f, 70, 60) and not painted(f, 110, 60)       # on frame 4 only the first link
    assert painted(o.draw(blank(), 20), 150, 60)


def test_labels_and_frame_are_left_alone():
    frame = blank()
    o = PO.PathOverlay(PO.from_tracks([track(123, A, 0)]), SHAPE)
    f0 = o.draw(frame, 0)
    assert (f0 != GREY).any() and not painted(f0, 50, 60, r=4)  # its ID, beside the box
    assert not (frame != GREY).any()                    # draw returns a copy
    assert not (PO.PathOverlay([], SHAPE).draw(frame, 3) != GREY).any()


def table(tracks, traj_of=None):
    rows = [{**d.as_dict(), "track_id": t.id} for t in tracks for d in t.dets]
    df = pd.DataFrame(rows)
    if traj_of is not None:
        df["trajectory_id"] = df.track_id.map(traj_of)
    return df


def test_from_table_takes_trajectories_where_they_exist():
    fall = track(0, [(100, 100), (110, 120), (120, 140), (130, 160)], 0)
    rise = track(1, [(138, 152), (146, 140), (154, 130)], 5)
    assert [p.id for p in PO.from_table(table([fall, rise]))] == [0, 1]        # tracks
    df = table([fall, rise], {0: 4, 1: 4})
    auto = PO.from_table(df)
    assert [p.id for p in auto] == [4] and list(auto[0].frames) == [0, 1, 2, 3, 5, 6, 7]
    assert len(PO.from_table(df, "track")) == 2
    try:
        PO.from_table(table([fall, rise]), "trajectory")
        raise AssertionError("no trajectory_id: should refuse")
    except ValueError:
        pass
    assert PO.from_table(pd.DataFrame()) == []
    # a table and its tracks give the same picture
    a = PO.PathOverlay(PO.from_table(df, "track"), SHAPE).draw(blank(), 20)
    b = PO.PathOverlay(PO.from_tracks([fall, rise]), SHAPE).draw(blank(), 20)
    assert np.array_equal(a, b)


def write_video(path, n=12):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 24.0, (SHAPE[1], SHAPE[0]))
    assert vw.isOpened()
    for _ in range(n):
        vw.write(blank())
    vw.release()


def read_video(path):
    cap, out = cv2.VideoCapture(str(path)), []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(f)
    cap.release()
    return out


def test_the_tool_end_to_end():
    import path_overlay as TOOL
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "clips").mkdir()
        tags = d / "work" / "tags" / "toy"
        tags.mkdir(parents=True)
        write_video(d / "clips" / "toy.mp4")
        table([track(3, A, 2)]).to_csv(tags / "detections.csv", index=False)
        pd.DataFrame([dict(ember_id=0, x=300.0, y=200.0, frame_start=4, frame_end=8,
                           looks=2, peak=90.0, area=9.0)]).to_csv(tags / "resting.csv", index=False)
        argv = ["--work", str(d / "work"), "--clips", str(d / "clips")]
        assert TOOL.main(argv) == 0
        out = tags / "overlay_paths.mp4"
        frames = read_video(out)
        assert len(frames) == 12 and frames[0].shape == SHAPE
        assert not (tags / "overlay_paths.part.mp4").exists()
        col = PO.colour(3)
        assert not is_colour(frames[2], 70, 60, col, tol=70)        # first seen on frame 2
        assert is_colour(frames[4], 70, 60, col, tol=70)            # a video codec: loose match
        assert is_colour(frames[11], 110, 60, col, tol=70)          # still there at the end
        pic = cv2.imread(str(tags / "overlay_paths.jpg"))
        assert pic.shape == SHAPE and is_colour(pic, 150, 60, col, tol=70)
        stamp = out.stat().st_mtime_ns
        assert TOOL.main(argv) == 0 and out.stat().st_mtime_ns == stamp     # skipped, not redone
        # part of the clip, half size, paths that leave, any CSV in the kit's columns
        half = d / "half.mp4"
        assert TOOL.main(["--csv", str(tags / "detections.csv"), "--video",
                          str(d / "clips" / "toy.mp4"), "--out", str(half), "--frames", "4-9",
                          "--scale", "0.5", "--linger", "1", "--no-ids"]) == 0
        frames = read_video(half)
        assert len(frames) == 6 and frames[0].shape == (120, 200, 3)
        assert is_colour(frames[0], 35, 30, col, tol=90)            # frame 4: the path so far
        assert not is_colour(frames[5], 35, 30, col, tol=90)        # frame 9: last seen 5, gone


def test_the_tag_stage_overlay_with_paths():
    """run_all's overlay writer (what a tag job runs), without a model."""
    import run_all
    fall = track(0, [(100, 100), (110, 120), (120, 140), (130, 160)], 0)
    rise = track(1, [(138, 152), (146, 140), (154, 130)], 5)
    rows, _ = TJ.build([fall, rise])
    cand = [det(2, 300, 50)]
    rest = [dict(ember_id=0, x=320.0, y=200.0, frame_start=1, frame_end=6)]
    with tempfile.TemporaryDirectory() as d:
        for name, traj in (("plain.mp4", None), ("paths.mp4", rows)):
            run_all.write_tag_overlay(Path(d) / name, [blank() for _ in range(10)], 24.0,
                                      SHAPE[:2], [fall, rise], cand, rest, traj)
        plain, paths = read_video(Path(d) / "plain.mp4"), read_video(Path(d) / "paths.mp4")
    assert len(plain) == len(paths) == 10

    def drawn(img, x, y, r=2, by=25):       # a thin slanted line through a video codec: loose
        return int(np.abs(img[y - r:y + r + 1, x - r:x + r + 1].astype(int) - GREY).max()) > by

    assert not drawn(plain[9], 115, 130)                            # boxes only: no path
    assert drawn(paths[9], 115, 130)                                # the fall, kept
    assert drawn(paths[9], 134, 156)                                # joined across the bounce
    assert drawn(plain[3], 320, 188) and drawn(paths[3], 320, 188)  # the resting ember's circle
    assert drawn(plain[2], 300, 50 - 6) and drawn(plain[0], 100, 100 - 6)   # dropped + tagged boxes


if __name__ == "__main__":
    for k, v in list(globals().items()):
        if k.startswith("test_"):
            v()
            print("ok ", k)
