#!/usr/bin/env python3
"""Checks for the linker (firebrand/track.py). No video, seconds.

    python tests/test_track.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from firebrand import track as T  # noqa: E402
from firebrand.config import TrackConfig  # noqa: E402
from firebrand.detect import Detection  # noqa: E402


def reference_link(dets_by_frame, cfg, step_ratio):
    """The linker as first written, double loop and all: the array version in
    track.link must give exactly this."""
    live, done, next_id = [], [], 0
    for fi in sorted(dets_by_frame):
        dets = dets_by_frame[fi]
        still, retired = [], []
        for t in live:
            (still if fi - t.last().frame <= cfg.max_gap + 1 else retired).append(t)
        live, done = still, done + retired
        if not dets:
            continue
        if not live:
            for d in dets:
                live.append(T.Track(next_id, [d]))
                next_id += 1
            continue
        C = np.full((len(live), len(dets)), 1e6, float)
        for i, t in enumerate(live):
            cands, step = t.predict(cfg, step_ratio, fi)
            gap = max(fi - t.last().frame, 1)
            gate = max(cfg.gate_frac * step * gap, 6.0)
            for j, d in enumerate(dets):
                dist = min(float(np.hypot(d.x - px, d.y - py)) for px, py in cands)
                if dist <= gate:
                    C[i, j] = dist
        ri, ci = linear_sum_assignment(C)
        used = set()
        for i, j in zip(ri, ci):
            if C[i, j] < 1e5:
                live[i].dets.append(dets[j])
                used.add(j)
        for j, d in enumerate(dets):
            if j not in used:
                live.append(T.Track(next_id, [d]))
                next_id += 1
    return done + live


def scene(seed, n_embers=40, n_noise=25, frames=30):
    rng = np.random.default_rng(seed)
    by = {f: [] for f in range(frames)}
    for _ in range(n_embers):
        x, y = rng.uniform(0, 1500, 2)
        vx, vy = rng.uniform(-40, 40, 2)
        f0 = int(rng.integers(0, frames - 5))
        for f in range(f0, min(frames, f0 + int(rng.integers(3, 15)))):
            if rng.random() < 0.9:                 # some frames missed: exercises gaps
                k = f - f0
                by[f].append(Detection(f, x + vx * k + rng.normal(0, 1), y + vy * k + rng.normal(0, 1),
                                       float(np.hypot(vx, vy)), 3.0, float(np.arctan2(vy, vx)),
                                       float(rng.uniform(20, 200)), 30, 900.0))
    for f in range(frames):
        for _ in range(int(rng.integers(0, n_noise))):
            by[f].append(Detection(f, *rng.uniform(0, 1500, 2), float(rng.uniform(3, 30)), 2.0,
                                   float(rng.uniform(-1.5, 1.5)), 30.0, 10, 100.0))
    by[7] = []                                     # an empty frame
    return by


def test_link_is_exactly_the_double_loop():
    cfg = TrackConfig()
    for seed in range(6):
        by = scene(seed)
        a, b = T.link(by, cfg, step_ratio=1.0), reference_link(by, cfg, 1.0)
        assert len(a) == len(b), (seed, len(a), len(b))
        for s, r in zip(a, b):
            assert s.id == r.id and [id(d) for d in s.dets] == [id(d) for d in r.dets], seed


def reference_stitch(tracks, cfg, max_frame_gap=4, pos_tol=1.4, ang_tol_deg=30.0):
    """track.stitch as first written: scans every track for every track."""
    order = sorted(tracks, key=lambda t: t.frames[0])
    used, out = set(), []
    for t in order:
        if id(t) in used:
            continue
        cur = t
        used.add(id(cur))
        merged = True
        while merged:
            merged = False
            end = cur.dets[-1]
            if len(cur.dets) < 2:
                break
            a, b = cur.dets[-2], cur.dets[-1]
            dt = max(b.frame - a.frame, 1)
            vx, vy = (b.x - a.x) / dt, (b.y - a.y) / dt
            speed = float(np.hypot(vx, vy))
            h0 = np.arctan2(vy, vx)
            best, best_d = None, np.inf
            for u in order:
                if id(u) in used:
                    continue
                gap = u.frames[0] - end.frame
                if not (1 <= gap <= max_frame_gap):
                    continue
                px, py = end.x + vx * gap, end.y + vy * gap
                d = float(np.hypot(u.dets[0].x - px, u.dets[0].y - py))
                if d > pos_tol * speed * gap:
                    continue
                if len(u.headings):
                    dh = abs(np.angle(np.exp(1j * (u.headings[0] - h0))))
                    if np.rad2deg(dh) > ang_tol_deg:
                        continue
                if d < best_d:
                    best, best_d = u, d
            if best is not None:
                cur.dets.extend(best.dets)
                cur.dets.sort(key=lambda d: d.frame)
                used.add(id(best))
                merged = True
        out.append(cur)
    return out


def test_stitch_is_exactly_the_full_scan():
    cfg = TrackConfig()
    for seed in range(6):
        by = scene(seed, n_embers=60)
        a = T.stitch(T.link(by, cfg, step_ratio=1.0), cfg)
        b = reference_stitch(T.link(by, cfg, step_ratio=1.0), cfg)
        assert [(t.id, [(d.frame, d.x) for d in t.dets]) for t in a] == \
               [(t.id, [(d.frame, d.x) for d in t.dets]) for t in b], seed
        assert len(a) < len(T.link(by, cfg, step_ratio=1.0)) or seed, "nothing stitched"


def streak(x, y, L, th=0.0, W=2.0, peak=200.0):
    return Detection(0, x, y, L, W, th, peak, 30, 900.0)


def test_merge_joins_the_dashes_of_one_streak():
    cfg = TrackConfig()
    out = T.merge_collinear([streak(100, 50, 20), streak(130, 50, 20), streak(160, 50.5, 20)], cfg)
    assert len(out) == 1, out
    d = out[0]
    assert abs(d.x - 130) < 1 and abs(d.L - 80) < 1 and d.area == 90, d


def test_merge_leaves_apart_what_is_apart():
    cfg = TrackConfig()
    for dets in (
        [streak(100, 50, 20), streak(100, 56, 20)],              # parallel neighbours
        [streak(100, 50, 20), streak(200, 50, 20)],              # same line, far apart
        [streak(100, 50, 20), streak(125, 50, 20, th=1.2)],      # crossing
        [streak(100, 50, 3, W=3.0), streak(106, 50, 3, W=3.0)],  # blobs: no direction
    ):
        assert len(T.merge_collinear(dets, cfg)) == 2, dets


def test_merge_treats_angle_as_mod_pi():
    cfg = TrackConfig()
    th = np.pi / 2 - 0.01
    a = streak(100, 100, 20, th=th)
    b = streak(100 + 30 * np.cos(th), 100 + 30 * np.sin(th), 20, th=th - np.pi)
    assert len(T.merge_collinear([a, b], cfg)) == 1


def test_merge_scales_with_resolution():
    near = [streak(100, 50, 20), streak(140, 50, 20)]            # 20 px between ends
    assert len(T.merge_collinear(near, TrackConfig())) == 2      # 1920 wide: too far
    assert len(T.merge_collinear(near, TrackConfig(max_step_px=240.0))) == 1   # 3840 wide


if __name__ == "__main__":
    for k, v in list(globals().items()):
        if k.startswith("test_"):
            v()
            print("ok ", k)
