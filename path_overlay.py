#!/usr/bin/env python3
"""Overlay video where every firebrand's whole path stays on screen.

    python3 path_overlay.py                          # every clip in work/tags
    python3 path_overlay.py --clip clip5 --scale 0.5
    python3 path_overlay.py --clip clip5 --frames 96-240 --linger 48
    python3 path_overlay.py --csv clips/clip5_manual.csv --video clips/clip5.mp4

On each frame a firebrand gets its usual green box, and everywhere it has been
tagged before stays drawn: its positions joined into one line in its own colour,
a dot per frame, its ID beside the box. One ember = one trajectory_id where the
tags have them (an ember across its bounces; trajectories.py writes them), else
one track_id. Cyan circles are resting embers, as on the tag overlay.

Reads work/tags/<clip>/detections.csv and the clip from --clips; writes, per clip,
    work/tags/<clip>/overlay_paths.mp4    the video
    work/tags/<clip>/overlay_paths.jpg    the last frame with every path, whole
A clip that already has its video is skipped (--force redoes it). With --csv, any
table in the kit's columns (frame, track_id, x, y, L, W, theta) is drawn on
--video: hand tags from tag_tool.py, or work/clips/<clip>/detections.csv for the
classical detector's tracks; the output goes beside the CSV as <name>_paths.mp4.

Runs on tags already made; no model, no GPU. A 4K frame takes a fraction of a
second, and a 4K video is a few hundred MB: --scale 0.5 writes 1080p instead.
The drawing itself is firebrand/pathoverlay.py; `--overlay-paths` on a detect or
tag job draws the same thing into that job's overlay.mp4.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from firebrand import detect as D, pathoverlay as PO  # noqa: E402
from firebrand.frames import VIDEO_EXTS  # noqa: E402
from firebrand.negatives import open_at  # noqa: E402
from firebrand.reallabels import read_csv_or_empty  # noqa: E402

CYAN = (255, 255, 0)


def parse_frames(spec):
    """'96-240' -> (96, 240); None -> (0, None): the whole clip."""
    if not spec:
        return 0, None
    a, sep, b = spec.partition("-")
    if not (a.isdigit() and sep and b.isdigit() and int(b) >= int(a)):
        raise SystemExit(f"--frames {spec!r}: want FIRST-LAST, e.g. 96-240")
    return int(a), int(b)


def find_video(clips, name):
    hits = sorted(p for p in Path(clips).glob(name + ".*") if p.suffix.lower() in VIDEO_EXTS) \
        if Path(clips).is_dir() else []
    return hits[0] if hits else None


def render(video, df, out, by="auto", linger=None, frames=(0, None), scale=1.0,
           labels=True, resting=None, log=print):
    """Draw the table's paths on the video -> `out` (+ a .jpg of every path whole).
    Returns (frames written, paths)."""
    paths = PO.from_table(df, by)
    first, end = frames                                 # end None: until the video runs out
    cap = open_at(video, first)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    # for the progress line only: the container's count can run a few frames long
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    stop = end if end is not None else total - 1 if total else None
    expect = stop - first + 1 if stop is not None else 0
    by_frame = {}
    for r in df.itertuples():                           # rows have .x .y .L .W .theta: all draw needs
        by_frame.setdefault(int(r.frame), []).append(r)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Written under another name and renamed at the end, so a run that is cut
    # off never leaves half a video that looks finished.
    part = out.with_name(out.stem + ".part" + out.suffix)
    painter = vw = prev = None
    n, t0, told = 0, time.time(), time.time()
    try:
        i = first
        while end is None or i <= end:
            ok, f = cap.read()
            if not ok:
                break
            if painter is None:
                H, W = f.shape[:2]
                size = (max(2, int(round(W * scale))), max(2, int(round(H * scale))))
                # lines at least 1 px wide in the video as written, whatever --scale is
                thick = max(int(round(max(1.0, W / 1920.0))), int(np.ceil(1.0 / scale)))
                painter = PO.PathOverlay(paths, (H, W), linger=linger, labels=labels,
                                         thickness=thick)
                vw = cv2.VideoWriter(str(part), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
                if not vw.isOpened():
                    raise SystemExit(f"cannot write {part} (no mp4v encoder in this OpenCV?)")
            prev = f
            g = D.draw(painter.draw(f, i), by_frame.get(i, []))
            for e in resting or ():
                if e["frame_start"] <= i <= e["frame_end"]:
                    cv2.circle(g, (int(e["x"]), int(e["y"])), 12, CYAN, 2)
            vw.write(g if scale == 1.0 else cv2.resize(g, size, interpolation=cv2.INTER_AREA))
            n += 1
            i += 1
            if time.time() - told > 15:
                told = time.time()
                log(f"    frame {n}{f'/{expect}' if expect > 0 else ''}  "
                    f"{n / (told - t0):.1f} fps")
    finally:
        cap.release()
        if vw is not None:
            vw.release()
    if not n:
        part.unlink(missing_ok=True)
        raise SystemExit(f"no frames read from {video} at frame {first}")
    os.replace(part, out)
    pic = PO.still(prev, paths, upto=first + n - 1, thickness=thick)
    cv2.imwrite(str(out.with_suffix(".jpg")),
                pic if scale == 1.0 else cv2.resize(pic, size, interpolation=cv2.INTER_AREA))
    return n, paths


def one(video, table, out, args, resting=None):
    out = Path(out)
    if out.exists() and not args.force:
        print(f"  {out}: already there, skipping (--force to redo)")
        return False
    df = read_csv_or_empty(table)
    if not len(df) or "track_id" not in df:
        print(f"  {table}: no tagged firebrands in it, nothing to draw")
        return False
    rest = read_csv_or_empty(resting) if resting else None
    rest = rest.to_dict("records") if rest is not None and len(rest) else []
    try:
        what = "trajectories" if PO.id_column(df, args.by) == "trajectory_id" else "tracks"
    except ValueError as e:
        raise SystemExit(f"{table}: {e}")
    t0 = time.time()
    n, paths = render(video, df, out, by=args.by, linger=args.linger,
                      frames=parse_frames(args.frames), scale=args.scale,
                      labels=not args.no_ids, resting=rest)
    print(f"  {out}: {n} frames, {len(paths)} {what}, {len(df)} tagged positions "
          f"({time.time() - t0:.0f}s; picture of every path: {out.with_suffix('.jpg').name})")
    return True


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default="work")
    ap.add_argument("--clips", default="clips", help="folder with the videos")
    ap.add_argument("--clip", action="append", help="only these clips (repeatable)")
    ap.add_argument("--csv", help="draw this table instead of work/tags (needs --video)")
    ap.add_argument("--video", help="the video --csv belongs to")
    ap.add_argument("--out", help="output video (one clip, or --csv)")
    ap.add_argument("--by", choices=("auto", "track", "trajectory"), default="auto",
                    help="what counts as one firebrand: its track, or its trajectory (one "
                         "ember across its bounces); auto = trajectories where the tags have them")
    ap.add_argument("--linger", type=int, default=None, metavar="N",
                    help="a path leaves N frames after its ember was last seen "
                         "(default: it stays to the end of the clip)")
    ap.add_argument("--frames", metavar="FIRST-LAST",
                    help="only this part of the clip; paths from before it are already drawn")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="size of the output video, e.g. 0.5 for 1080p from a 4K clip")
    ap.add_argument("--no-ids", action="store_true", help="no ID numbers beside the boxes")
    ap.add_argument("--force", action="store_true", help="redo videos that already exist")
    args = ap.parse_args(argv)
    if not 0.05 <= args.scale <= 1.0:
        raise SystemExit("--scale: between 0.05 and 1")

    if args.csv or args.video:
        if not (args.csv and args.video):
            raise SystemExit("--csv and --video go together")
        csv = Path(args.csv)
        one(args.video, csv, args.out or csv.with_name(csv.stem + "_paths.mp4"), args)
        return 0

    tags = Path(args.work) / "tags"
    names = args.clip or (sorted(p.name for p in tags.iterdir()
                                 if (p / "detections.csv").exists()) if tags.exists() else [])
    if not names:
        raise SystemExit(f"no tagged clips in {tags}/ -- pull them first:\n"
                         f"  bash scripts/05-gcs_sync.sh tags")
    if args.out and len(names) != 1:
        raise SystemExit("--out names one file: use it with one --clip")
    for name in names:
        video = find_video(args.clips, name)
        if video is None:
            print(f"  {name}: no video called {name}.* in {args.clips}/ -- skipped")
            continue
        d = tags / name
        one(video, d / "detections.csv", args.out or d / "overlay_paths.mp4", args,
            resting=d / "resting.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
