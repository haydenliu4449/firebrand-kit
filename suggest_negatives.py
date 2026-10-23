#!/usr/bin/env python3
"""Propose stretches of footage with NO firebrands, for work/negatives.txt.

    bash scripts/05-gcs_sync.sh review     # classical detections
    bash scripts/05-gcs_sync.sh tags       # optional: the model's tags
    python suggest_negatives.py            # writes work/negatives.suggested.txt

WHY. The model learns "firebrand" from synthetic embers painted onto your
footage, and every training scene has embers in it. It has never been shown a
scene with nothing in it, so when a clip goes quiet it tags whatever texture or
noise looks most streak-like. The fix is to train on real frames where you know
there is nothing: listed in work/negatives.txt, they become "nothing here"
training tiles, clean backgrounds for the synthetic embers, and the yardstick
the eval step uses to pick a threshold that does not invent tracks.

Each line gives the frames (what negatives.txt wants) and the same stretch as
times in the video (what VLC shows): frame = seconds x fps.

HOW. A stretch is proposed when the classical detector -- which is very
precise -- accepted no track anywhere in the frame for at least --min-seconds.
It can still miss faint embers, so these are SUGGESTIONS: watch each stretch
(the clip or the tag overlay) and copy only the ones you agree with into
work/negatives.txt. If the model tagged anything inside a stretch, that is
shown too: those are the spontaneous tags this is meant to remove.

A frame with even one faint, real ember in it must not go in negatives.txt --
it would teach the model that faint embers are background. When unsure, leave
the stretch out; a few confirmed seconds per clip are plenty.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def clock(sec):
    """seconds -> m:ss.s, what a video player shows"""
    return f"{int(sec // 60)}:{sec % 60:04.1f}"


def read_csv_or_empty(path: Path):
    """A CSV the kit wrote, or an empty table: a clip with no tracks used to
    get a file holding only a newline, which pandas will not read."""
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def frames_with(path: Path):
    df = read_csv_or_empty(path)
    return set(df["frame"].astype(int)) if "frame" in df and len(df) else set()


def tracks_in(path: Path, a: int, b: int):
    df = read_csv_or_empty(path)
    if not len(df) or "frame" not in df:
        return 0
    inside = df[(df.frame >= a) & (df.frame <= b)]
    return int(inside["track_id"].nunique()) if len(inside) else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default="work")
    ap.add_argument("--min-seconds", type=float, default=2.0,
                    help="shortest quiet stretch worth proposing")
    ap.add_argument("--pad", type=int, default=6,
                    help="frames trimmed off each end of a stretch (embers entering "
                         "or leaving view are faint)")
    args = ap.parse_args(argv)

    work = Path(args.work)
    clips = sorted(d for d in (work / "clips").iterdir()
                   if (d / "done.json").exists()) if (work / "clips").exists() else []
    if not clips:
        raise SystemExit(f"no detect output in {work}/clips -- run: bash scripts/05-gcs_sync.sh review")

    lines, total = [], 0
    for d in clips:
        n = int(json.load(open(d / "done.json"))["n_frames"])
        fps = 24.0
        if (d / "config.json").exists():
            fps = float(json.load(open(d / "config.json"))["track"].get("fps") or fps)
        busy = frames_with(d / "detections.csv")
        # The detector needs half a median window of context at each end, so
        # the first and last few frames say nothing either way.
        lo_ok, hi_ok = 12, n - 7
        min_len = int(round(args.min_seconds * fps))
        start = None
        for f in range(lo_ok, hi_ok + 2):
            quiet = f <= hi_ok and f not in busy
            if quiet and start is None:
                start = f
            elif not quiet and start is not None:
                a, b = start + args.pad, f - 1 - args.pad
                if b - a + 1 >= min_len:
                    m = tracks_in(work / "tags" / d.name / "detections.csv", a, b)
                    note = (f"{clock(a / fps)}-{clock((b + 1) / fps)} in the video "
                            f"({(b - a + 1) / fps:.1f} s), no classical track")
                    if m:
                        note += f"; the model tagged {m} track(s) here"
                    lines.append(f"{d.name:10s} {a}-{b:<12d} # {note}")
                    total += b - a + 1
                start = None

    out = work / "negatives.suggested.txt"
    header = ("# SUGGESTED firebrand-free stretches -- confirm each by watching it,\n"
              "# then copy the lines you agree with into work/negatives.txt\n"
              "# clip      frames (0-based, inclusive)\n")
    out.write_text(header + "\n".join(lines) + ("\n" if lines else ""))
    if not lines:
        print(f"No stretch of {args.min_seconds:g} s or more without a classical track.")
        print("Try --min-seconds 1, or pick quiet frames yourself while watching the clips.")
        return 0
    print("\n".join(lines))
    print(f"\n{len(lines)} stretch(es), {total} frames -> {out}")
    print("Watch each one (the clip, or work/tags/<clip>/overlay.mp4), then copy the")
    print("lines you agree with into work/negatives.txt and run:")
    print("  bash scripts/03-upload_inputs.sh")
    print("  bash scripts/04-submit_job.sh train -- --force")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
