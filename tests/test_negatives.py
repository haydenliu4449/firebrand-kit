#!/usr/bin/env python3
"""Checks for how firebrand-free footage is read and sampled (firebrand/negatives.py).

    python tests/test_negatives.py

No video, no GPU, a second or two. Run it after touching negatives.py.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from firebrand import negatives as NG  # noqa: E402


def spans(name, a, b, fps=24.0):
    return [NG.Span(name, f"/x/{name}.mp4", a, b, fps)]


def test_times():
    assert NG.parse_time("6:40") == 400
    assert NG.parse_time("1:02:03.5") == 3723.5
    assert NG.parse_time("120") is None
    for bad in ("6:75", "1:60:00"):
        try:
            NG.parse_time(bad)
        except SystemExit:
            continue
        raise AssertionError(f"{bad} accepted")


def test_read_specs():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "negatives.txt"
        p.write_text("# comment\nclip3 0-150\nclip5 400-520, 900-1010\n"
                     "quiet.mp4 0:00-6:40, 9:15-22:30   # times\nclip8\n")
        s = NG.read_specs(p)
    assert s["clip3"] == [("f", 0, 150)]
    assert s["clip5"] == [("f", 400, 520), ("f", 900, 1010)]
    assert s["quiet"] == [("t", 0.0, 400.0), ("t", 555.0, 1350.0)]
    assert s["clip8"] == [None]


def test_split_is_disjoint_and_stable():
    runs = NG.plan(spans("v", 0, 24 * 60 * 30))            # 30 minutes
    tr = NG.pick(runs, "train", 10 ** 9, run_len=16)
    ev = NG.pick(runs, "eval", 10 ** 9)
    A = {f for r in tr for f in range(r.a, r.b + 1)}
    B = {f for r in ev for f in range(r.a, r.b + 1)}
    assert not A & B, "training and held-out frames overlap"
    share = len(ev) / (len(ev) + len(tr))
    assert 0.2 < share < 0.3, share
    # Editing negatives.txt (a sub-range here) must not change any block's role.
    sub = {(r.a // NG.BLOCK): r.role for r in NG.plan(spans("v", 5000, 20000))}
    full = {(r.a // NG.BLOCK): r.role for r in runs}
    assert all(full[k] == v for k, v in sub.items())


def test_spread():
    """Any budget must cover the whole video, not its beginning."""
    runs = NG.plan(spans("v", 0, 24 * 60 * 30))
    n = 24 * 60 * 30
    for budget in (160, 640, 2000):
        cand = NG.pick(runs, "train", budget, run_len=16)[: budget // 16]
        starts = sorted(r.a for r in cand)
        gaps = [b - a for a, b in zip([0] + starts, starts + [n])]
        assert max(gaps) < 2.5 * n / len(cand), (budget, max(gaps), n / len(cand))
        assert all(r.n == 16 for r in cand)


def test_scarce_uses_whole_blocks():
    runs = NG.plan(spans("c", 10, 200))                     # ~6 blocks
    cand = NG.pick(runs, "train", 640, run_len=16)
    assert sum(r.n for r in cand) == sum(r.n for r in runs if r.role == "train")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
