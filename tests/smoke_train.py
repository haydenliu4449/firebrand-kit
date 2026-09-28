#!/usr/bin/env python3
"""Prove the training path works before you spend a GPU hour on it.

    python tests/smoke_train.py

Runs the entire torch half of the pipeline at toy scale -- ~60 seconds on CPU,
~15 on a GPU. Builds 60 tiny tiles, constructs the model, trains 2 epochs, saves
a checkpoint, reloads it, runs sliding-window inference, and turns the output
back into tracks.

Why this exists: a GPU machine bills from the moment it starts, so discovering
a broken import 20 minutes into your first real run costs money and momentum.
Run this first on every new machine -- and on Cloud Batch it is the `smoke` step
(`bash scripts/04-submit_job.sh smoke`), the first job to submit after building
the image. It also checks the two things that only break inside a container:
that the GPU is visible, and that training resumes after an interruption.

Each stage reports PASS or FAIL independently, so a failure tells you which
piece is broken rather than just handing you a traceback.
"""
from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

HERE = Path(__file__).resolve().parent
RESULTS = []


def stage(name):
    """Decorator: run a step, record PASS/FAIL, keep going."""
    def deco(fn):
        def wrapped(*a, **k):
            t0 = time.time()
            try:
                out = fn(*a, **k)
                RESULTS.append((name, True, f"{time.time() - t0:.1f}s", ""))
                return out
            except Exception as e:  # noqa: BLE001
                RESULTS.append((name, False, f"{time.time() - t0:.1f}s",
                                f"{type(e).__name__}: {e}"))
                print(f"\n--- {name} failed ---")
                traceback.print_exc()
                return None
        return wrapped
    return deco


@stage("torch imports and device")
def check_torch():
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  torch {torch.__version__}, device={dev}")
    if dev == "cuda":
        print(f"  gpu: {torch.cuda.get_device_name(0)}")
        free, total = torch.cuda.mem_get_info()
        print(f"  vram: {free/1e9:.1f} / {total/1e9:.1f} GB free")
    else:
        import os
        if os.environ.get("FB_STEP") == "smoke":
            # On Batch the whole point of this job is the GPU, so no GPU is a
            # failure, not a note -- the job must come back FAILED.
            raise RuntimeError(
                "no GPU visible inside the Batch container. Check that the job "
                "JSON has installGpuDrivers: true and an accelerator, and read "
                "the hardware record in work/logs/<job>/task0-hardware.json")
        print("  no GPU visible -- fine for this smoke test, slow for real training.")
        print("  If this machine has one, the driver is probably missing; check")
        print("  `nvidia-smi` in a terminal.")
    return dev


@stage("build toy tiles")
def build_tiles():
    from firebrand import Config
    from firebrand import dataset as DS

    bgp = HERE / "sample_frame.png"
    bg = cv2.imread(str(bgp))
    if bg is None:
        raise FileNotFoundError(f"{bgp} missing")
    bg = cv2.resize(bg, (640, 360))

    cfg = Config()
    cfg.track.fps = 15.0
    cfg.detect.tophat_ksize = 7
    cfg.synth.n_particles_per_frame = (4, 10)
    cfg.synth.speed_px = (8.0, 30.0)
    cfg.synth.reencode_h264 = False          # keep the smoke test fast
    cfg.train.tile = 128
    cfg.train.stride = 96

    X, Y, frames, gt = DS.build_from_synthetic(bg, 30, cfg, seed=0, max_tiles=60)
    if len(X) < 8:
        raise RuntimeError(f"only {len(X)} tiles built; expected dozens")
    print(f"  {len(X)} tiles, shape {X.shape[1:]}, "
          f"{DS.tile_stats(X, Y)['tiles_with_signal']:.0%} contain signal")
    return cfg, X, Y, frames


@stage("construct model")
def build_model(cfg):
    import torch
    from firebrand import model as M

    net = M.build(cfg.train)
    n = net.n_params
    print(f"  TinyUNet, {n/1e6:.2f}M parameters")
    x = torch.zeros(2, cfg.train.n_frames_stack, cfg.train.tile, cfg.train.tile)
    with torch.no_grad():
        y = net(x)
    want = (2, 1, cfg.train.tile, cfg.train.tile)
    if tuple(y.shape) != want:
        raise RuntimeError(f"output shape {tuple(y.shape)}, expected {want}")
    print(f"  forward pass {tuple(x.shape)} -> {tuple(y.shape)}")
    return net


@stage("loss is finite and decreases on one batch")
def check_loss(cfg, X, Y):
    """Overfit a single batch for a few steps. If the loss will not fall here,
    it will not fall on the real dataset either, and this takes seconds."""
    import torch
    from firebrand import model as M
    from firebrand.dataset import StreakTiles, suggested_pos_weight

    net = M.build(cfg.train)
    lossf = M.StreakLoss(suggested_pos_weight(Y), cfg.train.dice_weight)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)

    ds = StreakTiles(X[:4], Y[:4], train=False)
    xb = torch.stack([ds[i][0] for i in range(len(ds))])
    yb = torch.stack([ds[i][1] for i in range(len(ds))])

    first = last = None
    for i in range(12):
        opt.zero_grad()
        l = lossf(net(xb), yb)
        if not torch.isfinite(l):
            raise RuntimeError(f"loss is {l.item()} at step {i}")
        l.backward()
        opt.step()
        if i == 0:
            first = float(l)
        last = float(l)
    print(f"  loss {first:.4f} -> {last:.4f} over 12 steps")
    if last >= first:
        raise RuntimeError("loss did not decrease while overfitting 4 tiles -- "
                           "something is wrong with the loss or the targets")


@stage("train 2 epochs and checkpoint")
def train_short(cfg, X, Y, tmp):
    from firebrand import train as TR

    cfg.train.epochs = 2
    cfg.train.batch_size = 4
    model, hist = TR.train(X, Y, cfg.train, out_dir=tmp, log_every=1)
    ck = Path(tmp) / "best.pt"
    if not ck.exists():
        raise RuntimeError("no checkpoint written")
    print(f"  checkpoint {ck.stat().st_size/1e6:.1f} MB, "
          f"final val F1 {hist[-1]['val_f1']:.3f}")
    return model


@stage("resume an interrupted run from last.pt")
def resume_check(cfg, X, Y, tmp):
    """What a preempted Spot VM does: stop after epoch 1 of 3, start again,
    and check the second run continues at epoch 2 instead of starting over."""
    import json
    from firebrand import train as TR

    d = Path(tmp) / "resume"
    cfg.train.epochs = 3
    cfg.train.batch_size = 4

    class Stop(Exception):
        pass

    real_save = TR._save_atomic
    calls = {"n": 0}

    def save_then_die(obj, path):
        real_save(obj, path)
        if str(path).endswith("last.pt"):
            calls["n"] += 1
            if calls["n"] == 2:            # epochs 0 and 1 done -> "preempted"
                raise Stop()

    TR._save_atomic = save_then_die
    try:
        TR.train(X, Y, cfg.train, out_dir=d, log_every=1)
        raise RuntimeError("the simulated interruption did not happen")
    except Stop:
        pass
    finally:
        TR._save_atomic = real_save
    if (d / "done.json").exists():
        raise RuntimeError("done.json written by an interrupted run")

    _, hist = TR.train(X, Y, cfg.train, out_dir=d, log_every=1)
    epochs = [h["epoch"] for h in hist]
    if epochs != [0, 1, 2]:
        raise RuntimeError(f"history after resume is {epochs}, expected [0, 1, 2]")
    if not (d / "done.json").exists():
        raise RuntimeError("done.json missing after the resumed run finished")
    print(f"  interrupted after epoch 1, resumed, finished epoch 2: "
          f"{json.load(open(d / 'done.json'))}")


@stage("reload checkpoint")
def reload_ck(tmp, dev="cpu"):
    from firebrand import train as TR
    model, cfg = TR.load(Path(tmp) / "best.pt", device=dev)
    print(f"  reloaded, {model.n_params/1e6:.2f}M parameters")
    return model


@stage("sliding-window inference")
def infer(cfg, model, frames, dev="cpu"):
    from firebrand import infer as I
    from firebrand.dataset import residual_stacks
    from firebrand.frames import ArraySource

    got = None
    for i, stack in residual_stacks(ArraySource(frames), cfg.detect, None,
                                    cfg.train.n_frames_stack):
        got = (i, stack)
        break
    if got is None:
        raise RuntimeError("no residual stacks produced")

    i, stack = got
    prob = I.predict_map(model, stack, cfg.train.tile, cfg.train.stride, dev)
    if prob.shape != stack.shape[1:]:
        raise RuntimeError(f"prob map {prob.shape} != frame {stack.shape[1:]}")
    if not np.isfinite(prob).all():
        raise RuntimeError("probability map contains non-finite values")
    dets = I.detections_from_map(prob, i, cfg, thresh=0.5)
    print(f"  prob map {prob.shape}, range [{prob.min():.3f}, {prob.max():.3f}], "
          f"{len(dets)} detections at threshold 0.5")
    print("  (a 2-epoch model detecting little or nothing is expected -- this")
    print("   stage checks the plumbing, not the accuracy)")


@stage("model output -> tracks")
def to_tracks(cfg, model, frames, dev="cpu"):
    from firebrand import infer as I
    from firebrand.frames import ArraySource
    dets, tracks = I.run_clip(model, ArraySource(frames[:14]), cfg, None,
                              thresh=0.5, device=dev)
    print(f"  ran {len(dets)} frames through the full model path, "
          f"{len(tracks)} accepted tracks")


def main():
    import tempfile
    print("firebrand training smoke test\n" + "=" * 46)

    dev = check_torch()
    if dev is None:
        print("\nThe torch check failed (see above). Either torch is not")
        print("importable -- `pip install torch`; the Batch image installs it --")
        print("or, on Batch, the container cannot see the GPU.")
        summary()
        return 1

    built = build_tiles()
    if built is None:
        summary()
        return 1
    cfg, X, Y, frames = built

    net = build_model(cfg)
    if net is not None:
        check_loss(cfg, X, Y)

    with tempfile.TemporaryDirectory() as tmp:
        model = train_short(cfg, X, Y, tmp)
        if model is not None:
            resume_check(cfg, X, Y, tmp)
            reloaded = reload_ck(tmp, dev)
            if reloaded is not None:
                infer(cfg, reloaded, frames, dev)
                to_tracks(cfg, reloaded, frames, dev)

    return summary()


def summary():
    print("\n" + "=" * 46)
    ok = True
    for name, passed, secs, err in RESULTS:
        print(f"  {'PASS' if passed else 'FAIL'}  {name:38s} {secs:>7s}")
        if err:
            print(f"        {err}")
        ok &= passed
    print("=" * 46)
    if ok:
        print("\nAll stages passed. The training path works on this machine.")
        print("On Cloud Batch the next step is:  bash scripts/04-submit_job.sh detect")
        return 0
    print("\nSomething is broken. Fix the first FAIL above -- later stages")
    print("depend on earlier ones, so one root cause often fails several.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
