"""Training loop.

Colab notes: a run on ~8k tiles of 256px with base_ch=16 fits comfortably in a
free-tier T4 session at batch_size 16, and takes roughly 15-25 minutes for 30
epochs. On CPU the same run is a few hours -- reduce tile to 192 and epochs to
12 if that is what you have.

The validation metric here is not loss. Loss on a 1-in-1500 class is dominated
by the background and will keep improving while detection quality does not.
Watch `val_f1` (pixel-level, at the operating threshold) and, more importantly,
run `evaluate.py` on a held-out clip -- pixel F1 still does not tell you whether
tracks come out intact.

RESUMABLE. Every epoch writes `last.pt` (model + optimiser + schedule + history)
next to `best.pt`. Call train() again with the same out_dir and it continues from
the last finished epoch instead of starting over. That is what makes Spot VMs
safe on Cloud Batch: a preempted task is retried and picks up where it stopped.
`done.json` is written only when every epoch has run, and it -- not best.pt --
is what callers should treat as "training finished". best.pt exists after the
first epoch, so treating it as the finish line would skip the rest of a run
that was interrupted at epoch 1.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import TrainConfig
from .dataset import StreakTiles, suggested_pos_weight
from .model import StreakLoss, build


def split(n, val_frac=0.15, seed=0):
    """Split `n` tiles into train/val INDEX ARRAYS.

    Returns indices, not slices of the data. Returning `X[t], Y[t]` instead --
    which is what this used to do -- makes a second full copy of the dataset:
    7,400 tiles of 3x256x256 is 1.45 GB for X plus 1.94 GB for float masks, so
    the split alone pushed a 15 GB VM to roughly 7 GB of live arrays before the
    first batch. StreakTiles takes `idx` and indexes lazily instead.

    NOTE: this splits tiles at random, which leaks between overlapping tiles
    from the same frame. It is fine for watching convergence, but for a number
    you would put in a paper, hold out whole *clips* instead and evaluate with
    evaluate.report().
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(int(n))
    n_val = int(n * val_frac)
    return idx[n_val:], idx[:n_val]


@torch.no_grad()
def pixel_scores(model, loader, device, thresh=0.5):
    model.eval()
    tp = fp = fn = 0
    loss_sum, n = 0.0, 0
    lossf = StreakLoss().to(device)
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss_sum += float(lossf(logits, yb)) * len(xb)
        n += len(xb)
        p = (torch.sigmoid(logits) > thresh)
        t = yb > 0.2
        tp += int((p & t).sum()); fp += int((p & ~t).sum()); fn += int((~p & t).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return dict(loss=loss_sum / max(n, 1), precision=prec, recall=rec,
                f1=2 * prec * rec / (prec + rec) if prec + rec else 0.0)


def default_workers(min_shm_bytes=1 << 30):
    """DataLoader workers that will not crash this machine.

    Workers hand batches to the main process through /dev/shm, and Docker gives
    a container only 64 MB of it unless told otherwise -- a few batches of
    16x3x256x256 float32 exceed that and the run dies with "bus error" or
    "insufficient shared memory" minutes in. The Batch job templates pass
    --shm-size, but if that ever gets lost this falls back to loading in the
    main process (slower, but it finishes). Windows has no /dev/shm and spawns
    rather than forks, which would copy the whole tile array into every worker,
    so it gets 0 as well.
    """
    if sys.platform.startswith("win"):
        return 0
    try:
        shm = shutil.disk_usage("/dev/shm").total
    except OSError:
        shm = 0
    if shm < min_shm_bytes:
        print(f"  /dev/shm is {shm / 1e6:.0f} MB -- loading data in the main "
              f"process (num_workers=0). Pass --shm-size to docker to fix.",
              flush=True)
        return 0
    return max(0, min(4, (os.cpu_count() or 2) - 1))


def _worker_init(_worker_id):
    """Give every DataLoader worker, in every epoch, its own augmentation RNG.

    Without this each worker is forked with a copy of the parent's generator,
    and the parent's copy never advances (it never draws a sample), so every
    worker in every epoch replays the same flips, rotations and noise. torch
    seeds each worker with base_seed + worker_id, and base_seed changes per
    epoch, so deriving from it gives fresh augmentation every time.
    """
    info = torch.utils.data.get_worker_info()
    if info is not None and hasattr(info.dataset, "rng"):
        info.dataset.rng = np.random.default_rng(torch.initial_seed() % (2 ** 32))


def _fingerprint(X, Y, cfg):
    """Enough to tell whether a last.pt belongs to this dataset and schedule."""
    return dict(n_tiles=int(len(X)), tile_shape=list(X.shape[1:]),
                epochs=int(cfg.epochs), batch_size=int(cfg.batch_size),
                seed=int(cfg.seed))


def _save_atomic(obj, path):
    """Write then rename, so a VM that dies mid-write leaves the previous file
    intact instead of a truncated one that fails to load on resume."""
    tmp = Path(str(path) + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def train(X, Y, cfg: TrainConfig, out_dir="runs/streak", device=None,
          auto_pos_weight=True, log_every=1, num_workers=None, heartbeat=50,
          resume=True):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if num_workers is None:
        num_workers = default_workers()

    # Resume state is read BEFORE pos_weight is recomputed, so a resumed run
    # keeps the loss it started with.
    last_path = out / "last.pt"
    fp = _fingerprint(X, Y, cfg)
    ck = None
    if resume and last_path.exists():
        ck = torch.load(last_path, map_location=device, weights_only=False)
        if ck.get("fingerprint") != fp:
            print(f"  {last_path} is from a different dataset or schedule "
                  f"({ck.get('fingerprint')} vs {fp}) -- starting fresh", flush=True)
            ck = None

    if ck is not None:
        cfg.pos_weight = float(ck["cfg"]["pos_weight"])
    elif auto_pos_weight:
        cfg.pos_weight = suggested_pos_weight(Y)

    ti, vi = split(len(X), cfg.val_frac, cfg.seed)
    dl_t = DataLoader(StreakTiles(X, Y, train=True, seed=cfg.seed, idx=ti),
                      batch_size=cfg.batch_size, shuffle=True,
                      num_workers=num_workers,
                      worker_init_fn=_worker_init if num_workers else None,
                      pin_memory=(device == "cuda"),
                      drop_last=len(ti) > cfg.batch_size)
    dl_v = DataLoader(StreakTiles(X, Y, train=False, idx=vi),
                      batch_size=cfg.batch_size)

    model = build(cfg).to(device)
    lossf = StreakLoss(cfg.pos_weight, cfg.dice_weight).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    use_amp = bool(cfg.amp and device == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    hist, best, start = [], -1.0, 0
    if ck is not None:
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        if use_amp and ck.get("scaler"):
            scaler.load_state_dict(ck["scaler"])
        hist, best, start = list(ck["hist"]), float(ck["best"]), int(ck["epoch"]) + 1
        torch.set_rng_state(ck["torch_rng"].cpu())
        print(f"  resuming from {last_path}: {start}/{cfg.epochs} epochs done, "
              f"best val F1 so far {best:.3f}", flush=True)

    nbytes = (X.nbytes + Y.nbytes) / 1e9
    print(f"device={device}  params={model.n_params/1e6:.2f}M  "
          f"train={len(ti)} val={len(vi)}  pos_weight={cfg.pos_weight:.1f}  "
          f"workers={num_workers}  tiles={nbytes:.2f}GB resident", flush=True)

    for ep in range(start, cfg.epochs):
        model.train()
        t0, run, n = time.time(), 0.0, 0
        # Heartbeat inside the epoch. An epoch here is a few hundred steps and
        # can run for minutes; a Jupyter proxy that sees no output for that long
        # will drop the websocket and the notebook reports a dead kernel even
        # though training is fine. Printing every `heartbeat` steps keeps the
        # connection alive and tells you the run is progressing, not wedged.
        for step, (xb, yb) in enumerate(dl_t):
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = lossf(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            run += float(loss) * len(xb); n += len(xb)
            if heartbeat and step and step % heartbeat == 0:
                print(f"    ep{ep} step {step}/{len(dl_t)} "
                      f"loss {run/max(n,1):.4f}", flush=True)
        sched.step()

        v = pixel_scores(model, dl_v, device)
        rec = dict(epoch=ep, train_loss=run / max(n, 1),
                   secs=round(time.time() - t0, 1), **{f"val_{k}": v[k] for k in v})
        hist.append(rec)
        if ep % log_every == 0 or ep == cfg.epochs - 1:
            print(f"  ep{ep:3d}  train {rec['train_loss']:.4f}  "
                  f"val {rec['val_loss']:.4f}  P {v['precision']:.3f} "
                  f"R {v['recall']:.3f}  F1 {v['f1']:.3f}  ({rec['secs']}s)",
                  flush=True)

        if v["f1"] > best:
            best = v["f1"]
            _save_atomic({"model": model.state_dict(), "cfg": dict(cfg.__dict__),
                          "val_f1": best, "epoch": ep}, out / "best.pt")
        # Written every epoch, not at the end: if the run is killed you still
        # have the curve up to that point, best.pt is already on disk, and
        # last.pt lets the next attempt continue from here.
        with open(out / "history.json", "w") as fh:
            json.dump(hist, fh, indent=2)
        _save_atomic({"model": model.state_dict(), "opt": opt.state_dict(),
                      "sched": sched.state_dict(),
                      "scaler": scaler.state_dict() if use_amp else None,
                      "epoch": ep, "hist": hist, "best": best,
                      "cfg": dict(cfg.__dict__), "fingerprint": fp,
                      "torch_rng": torch.get_rng_state()}, last_path)

    with open(out / "done.json", "w") as fh:
        json.dump({"epochs": cfg.epochs, "best_val_f1": best,
                   "finished": time.strftime("%Y-%m-%d %H:%M:%S")}, fh, indent=2)
    print(f"best val pixel-F1 {best:.3f} -> {out/'best.pt'}", flush=True)
    return model, hist


def load(checkpoint, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = TrainConfig(**ck["cfg"]) if isinstance(ck["cfg"], dict) else ck["cfg"]
    model = build(cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, cfg
