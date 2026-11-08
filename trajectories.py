#!/usr/bin/env python3
"""Join tagged tracks into trajectories: one ember across its hops, and where it landed.

    python3 trajectories.py                         # every clip in work/tags
    python3 trajectories.py --clip clip5 --views 12 --clips work/recall_review/clips

Reads work/tags/<clip>/detections.csv (+ resting.csv) and writes, per clip (with
--calib, default calib/left_front.json, also ground positions in metres):
    trajectories.csv          one row per trajectory: its tracks, hops, contacts, landing
    detections.csv            gains a trajectory_id column (the file is rewritten)
    paths/traj_<id>.jpg       (--views N) the N trajectories with most hops, drawn on the
                              brightest pixel of each of their frames: look at these before
                              trusting a trajectory from a dense shower
Runs on tags already made; no model, no GPU. See firebrand/trajectory.py for the rules.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from firebrand import detect as D, track as T, trajectory as TJ  # noqa: E402
from firebrand.calibration import GroundCalibration  # noqa: E402
from firebrand.reallabels import read_csv_or_empty  # noqa: E402


def tracks_from(df):
    out = []
    for tid, g in df.sort_values("frame").groupby("track_id"):
        dets = [D.Detection(int(r.frame), r.x, r.y, r.L, r.W, r.theta, r.peak, int(r.area), r.flux)
                for r in g.itertuples()]
        out.append(T.Track(int(tid), dets))
    return out


def path_view(video, tracks, row, out_path, pad=60):
    """Max over the trajectory's frames, cropped round it, boxes + contacts drawn."""
    dets = [d for t in tracks for d in t.dets]
    x0 = int(max(min(d.x for d in dets) - pad, 0))
    y0 = int(max(min(d.y for d in dets) - pad, 0))
    x1 = int(max(d.x for d in dets) + pad)
    y1 = int(max(d.y for d in dets) + pad)
    frames = sorted({d.frame for d in dets})
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frames[0])
    acc = None
    for f in range(frames[0], frames[-1] + 1):
        ok, im = cap.read()
        if not ok:
            break
        c = im[y0:y1, x0:x1]
        acc = c.copy() if acc is None else np.maximum(acc, c)
    cap.release()
    if acc is None:
        return
    sc = 2 if max(acc.shape[:2]) < 700 else 1
    acc = cv2.resize(acc, None, fx=sc, fy=sc, interpolation=cv2.INTER_NEAREST)
    colours = [(0, 255, 0), (255, 200, 0), (0, 200, 255), (255, 0, 255)]
    for k, t in enumerate(tracks):
        for d in t.dets:
            p = (D.box_points(d, pad=2) - [x0, y0]) * sc
            cv2.polylines(acc, [p.astype(np.int32)], True, colours[k % 4], 1)
    for c in filter(None, str(row["contacts"]).split(";")):
        f, xy = c.split(":")
        x, y = map(float, xy.split(","))
        cv2.circle(acc, (int((x - x0) * sc), int((y - y0) * sc)), 9, (0, 0, 255), 2)
    cv2.putText(acc, f"traj {row['trajectory_id']}  hops {row['n_hops']}  frames "
                     f"{row['frame_start']}-{row['frame_end']}", (6, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imwrite(str(out_path), acc)


def run_clip(work, name, args):
    d = work / "tags" / name
    df = read_csv_or_empty(d / "detections.csv")
    if not len(df) or "track_id" not in df:
        return None
    rest = read_csv_or_empty(d / "resting.csv")
    rest = rest.to_dict("records") if len(rest) else []
    tracks = tracks_from(df)
    width = float(json.load(open(d / "done.json"))["shape"][1]) if (d / "done.json").exists() else 3840.0
    cal = GroundCalibration.load(args.calib) if args.calib and Path(args.calib).exists() else None
    rows = TJ.write_clip(d, tracks, rest, width, cal=cal)
    if args.views:
        video = next((p for p in Path(args.clips).glob(name + ".*")), None)
        if video is not None:
            (d / "paths").mkdir(exist_ok=True)
            by = {t.id: t for t in tracks}
            for r in sorted(rows, key=lambda r: (-r["n_hops"], -r["n_det"]))[:args.views]:
                ts = [by[int(i)] for i in str(r["track_ids"]).split()]
                path_view(video, ts, r, d / "paths" / f"traj_{r['trajectory_id']}.jpg")
    n = len(rows)
    hop = sum(r["n_hops"] > 0 for r in rows)
    land = sum(r["landed_resting_id"] >= 0 for r in rows)
    if cal is not None:
        rp = d / "resting.csv"
        rr = read_csv_or_empty(rp)
        if len(rr):
            g = cal.to_ground(rr[["x", "y"]].to_numpy(float))
            rr["x_m"], rr["y_m"] = g[:, 0].round(3), g[:, 1].round(3)
            rr["ground_supported"] = cal.supported(rr[["x", "y"]].to_numpy(float)).astype(int)
            rr.to_csv(rp, index=False)
    print(f"  {name}: {len(tracks)} tracks -> {n} trajectories; {hop} with hops "
          f"({sum(r['n_hops'] for r in rows)} contacts); {land} end at a resting ember")
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--work", default="work")
    ap.add_argument("--clip", action="append")
    ap.add_argument("--views", type=int, default=0)
    ap.add_argument("--clips", default="clips")
    ap.add_argument("--calib", default="calib/left_front.json",
                    help="ground calibration (pixels -> metres); '' to skip")
    args = ap.parse_args()
    work = Path(args.work)
    names = args.clip or sorted(p.name for p in (work / "tags").iterdir() if p.is_dir())
    for n in names:
        run_clip(work, n, args)


if __name__ == "__main__":
    main()
