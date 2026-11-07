"""Trajectories: one ember from its first frame to where it comes to rest.

The tracker cuts a track at any sharp turn -- that is what keeps a link between
two different embers from passing as one -- so a saltating ember comes out as
pieces: descent, contact, ascent, descent... (approach.md, "A split at a sharp
turn"). And an ember that lands and lies still leaves the tracks entirely: the
rolling median takes it in, and only the resting pass (resting.py) finds it.

This joins the pieces back into trajectories, with rules a bounce must satisfy
and a chance meeting of two embers rarely does:
  * track A ends going DOWN the image, track B starts going UP;
  * B starts 1..max_gap frames after A ends, close to where A ended;
  * B keeps A's horizontal direction and is no faster than A (an impact loses
    speed), within a margin;
and the trajectory's end is tied to a resting ember that appears where and when
it ended. Contacts are where the hops' lengths will be measured, once pixels are
metres. Works on detections.csv, so it runs on tags already made.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TrajConfig:
    max_gap: int = 2            # frames from the end of a descent to the start of the rise
    near_px: float = 6.0        # px at 1920 wide, added to what the speeds allow
    speed_ratio: float = 1.1    # rise speed may exceed fall speed by this much (noise)
    reverse_frac: float = 0.1   # horizontal velocity may flip by this fraction of speed
    min_vertical: float = 0.2   # a fall and a rise must each move up/down by this share of
                                # their speed: a near-level ember wobbling by a pixel is no bounce
    land_px: float = 8.0        # px at 1920 wide: end of trajectory to a resting ember
    land_frames: int = 36       # a resting ember first seen up to this many frames later


def _vel(dets, at_end):
    """Velocity (px/frame) at a track's end or start, from its two outer points."""
    if len(dets) < 2:
        return None
    a, b = (dets[-2], dets[-1]) if at_end else (dets[0], dets[1])
    dt = max(b.frame - a.frame, 1)
    return np.array([(b.x - a.x) / dt, (b.y - a.y) / dt])


def hop_pairs(tracks, cfg: TrajConfig, scale=1.0):
    """[(i, j, contact)] -- track i's end is a contact and track j its rise. Each end
    and each start is used once, best match (nearest) first."""
    ends, starts = {}, {}
    for i, t in enumerate(tracks):
        v = _vel(t.dets, True)
        if v is not None and v[1] > cfg.min_vertical * np.hypot(*v):     # going down
            ends.setdefault(t.dets[-1].frame, []).append(i)
        v = _vel(t.dets, False)
        if v is not None and -v[1] > cfg.min_vertical * np.hypot(*v):    # going up
            starts.setdefault(t.dets[0].frame, []).append(i)
    cand = []
    for f, ids in ends.items():
        for i in ids:
            a = tracks[i].dets[-1]
            va = _vel(tracks[i].dets, True)
            sa = float(np.hypot(*va))
            for g in range(1, cfg.max_gap + 1):
                for j in starts.get(f + g, ()):
                    if j == i:
                        continue
                    b = tracks[j].dets[0]
                    vb = _vel(tracks[j].dets, False)
                    sb = float(np.hypot(*vb))
                    if sb > cfg.speed_ratio * sa + 1.0:
                        continue
                    if va[0] * vb[0] < -cfg.reverse_frac * sa * sb:
                        continue
                    d = float(np.hypot(b.x - a.x, b.y - a.y))
                    if d > (sa + sb) * g * 0.75 + cfg.near_px * scale:
                        continue
                    cand.append((d, i, j))
    used_i, used_j, out = set(), set(), []
    for d, i, j in sorted(cand):
        if i in used_i or j in used_j:
            continue
        used_i.add(i)
        used_j.add(j)
        a = tracks[i].dets[-1]
        out.append((i, j, (a.frame, a.x, a.y)))
    return out


def build(tracks, resting=None, cfg: TrajConfig | None = None, scale=1.0):
    """tracks (with .id, .dets in frame order) -> (rows, traj_of_track).

    rows: one dict per trajectory -- ids of its tracks in order, contacts, the
    resting ember it ended at (if any) and simple measures in pixels.
    traj_of_track: {track id: trajectory id}."""
    cfg = cfg or TrajConfig()
    pairs = hop_pairs(tracks, cfg, scale)
    nxt = {i: (j, c) for i, j, c in pairs}
    has_prev = {j for _i, j, _c in pairs}
    rows, traj_of = [], {}
    rest = list(resting) if resting is not None else []
    used_rest = set()
    for i in sorted(range(len(tracks)), key=lambda k: tracks[k].dets[0].frame):
        if i in has_prev:
            continue
        chain, contacts = [i], []
        while chain[-1] in nxt:
            j, c = nxt[chain[-1]]
            chain.append(j)
            contacts.append(c)
        tid = len(rows)
        dets = [d for k in chain for d in tracks[k].dets]
        for k in chain:
            traj_of[tracks[k].id] = tid
        last = tracks[chain[-1]].dets[-1]
        landed = None
        best = cfg.land_px * scale
        for r in rest:
            if r["ember_id"] in used_rest:
                continue
            if not (last.frame - 12 <= r["frame_start"] <= last.frame + cfg.land_frames):
                continue
            dd = float(np.hypot(r["x"] - last.x, r["y"] - last.y))
            if dd <= best:
                best, landed = dd, r["ember_id"]
        if landed is not None:
            used_rest.add(landed)
        path = float(sum(np.hypot(b.x - a.x, b.y - a.y) for a, b in zip(dets, dets[1:])))
        rows.append(dict(
            trajectory_id=tid, track_ids=" ".join(str(tracks[k].id) for k in chain),
            n_tracks=len(chain), n_hops=len(contacts), n_det=len(dets),
            frame_start=dets[0].frame, frame_end=last.frame,
            x0=dets[0].x, y0=dets[0].y, x1=last.x, y1=last.y,
            path_px=path,
            contacts=";".join(f"{f}:{x:.1f},{y:.1f}" for f, x, y in contacts),
            landed_resting_id=-1 if landed is None else int(landed)))
    return rows, traj_of


def write_clip(out_dir, tracks, resting_rows, width, cfg: TrajConfig | None = None):
    """Trajectories for one tagged clip: writes trajectories.csv and adds a
    trajectory_id column to detections.csv. Shared by the tag stage and by
    trajectories.py, so tags made either way carry the same thing. Returns rows."""
    import pandas as pd
    from pathlib import Path
    out_dir = Path(out_dir)
    rows, traj_of = build(tracks, resting_rows, cfg, scale=width / 1920.0)
    pd.DataFrame(rows, columns=None if rows else ["trajectory_id"]) \
        .to_csv(out_dir / "trajectories.csv", index=False)
    det = out_dir / "detections.csv"
    try:
        df = pd.read_csv(det)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return rows
    if len(df) and "track_id" in df:
        df["trajectory_id"] = df.track_id.astype(int).map(traj_of)
        df.to_csv(det, index=False)
    return rows
