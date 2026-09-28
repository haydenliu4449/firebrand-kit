#!/usr/bin/env python3
"""Turn one pipeline step into a Batch job file. Called by 02 and 04.

    python scripts/render_job.py detect --tasks 10 -- --max-frames 300

Reads IMAGE_URI, BUCKET_PATH, machine settings etc. from the environment
(scripts/common.sh exports them from .env) and fills templates/job.<gpu|cpu>.
template.json. Anything after `--` is passed to run_all.py unchanged.

The steps below are the whole menu. Each one is a single run_all.py invocation
on one kind of machine; nothing else happens in a job.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

STEPS = {
    "smoke": dict(
        profile="gpu", args=["smoke"], hours=0.75, per_clip=False,
        what="prove the image, GPU, training and resume work (~5 min, ~$0.10)"),
    "hardware": dict(
        profile="gpu", args=["hardware"], hours=0.5, per_clip=False,
        what="record what machine Batch gave you (the lab template's example)"),
    "detect": dict(
        profile="cpu", args=["run", "--stage", "detect"], hours=4, per_clip=True,
        what="classical detector on every clip -> tracks + contact sheets"),
    "tiles": dict(
        profile="cpu", args=["run", "--stage", "tiles"], hours=3, per_clip=False,
        what="fit the generator to verified tracks -> training tiles"),
    "train": dict(
        profile="gpu", args=["run", "--stage", "tiles,train,eval"], hours=6,
        per_clip=False,
        what="tiles (if not built) -> train -> evaluate against the classical detector"),
    "tag": dict(
        profile="gpu", args=["run", "--stage", "tag"], hours=6, per_clip=True,
        what="the trained model tags every clip -> work/tags/<clip>/"),
}

NAME_RE = re.compile(r"^[a-z]([a-z0-9-]{0,61}[a-z0-9])?$")


def env(key, default=None):
    v = os.environ.get(key, default)
    if v is None or v == "":
        sys.exit(f"{key} is not set -- run this through scripts/02 or 04, "
                 f"which load .env")
    return v


def render(step, name, tasks=1, extra=(), hours=None, provisioning=None):
    s = STEPS[step]
    tpl = ROOT / "templates" / f"job.{s['profile']}.template.json"
    job = json.loads(tpl.read_text(encoding="utf-8"))

    tg = job["taskGroups"][0]
    spec = tg["taskSpec"]
    c = spec["runnables"][0]["container"]
    c["imageUri"] = env("IMAGE_URI")
    # The whole command line is spelled out in the job rather than left to the
    # image's ENTRYPOINT: with only arguments in `commands`, Batch ran the first
    # argument ("smoke") as the program and the task died with exit 127.
    # `entrypoint` + full `commands` works whichever way Batch treats the
    # image's own ENTRYPOINT.
    c["entrypoint"] = "/opt/venv/bin/python"
    c["commands"] = ["/app/batch_entry.py"] + list(s["args"]) + list(extra)

    spec["maxRunDuration"] = f"{int(float(hours or s['hours']) * 3600)}s"
    spec["environment"]["variables"].update({
        "FB_JOB_NAME": name, "FB_STEP": step, "FB_MASK": env("MASK", "left_front.json")})
    spec["volumes"][0]["gcs"]["remotePath"] = env("BUCKET_PATH")

    tasks = max(1, int(tasks)) if s["per_clip"] else 1
    par_key = "GPU_PARALLEL" if s["profile"] == "gpu" else "CPU_PARALLEL"
    tg["taskCount"] = tasks
    tg["parallelism"] = max(1, min(tasks, int(env(par_key, "1"))))

    pol = job["allocationPolicy"]["instances"][0]["policy"]
    pol["provisioningModel"] = provisioning or env("PROVISIONING", "STANDARD")
    if s["profile"] == "gpu":
        pol["machineType"] = env("GPU_MACHINE")
        pol["accelerators"][0]["type"] = env("GPU_TYPE")
    else:
        pol["machineType"] = env("CPU_MACHINE")

    job["labels"] = {"app": "firebrand", "step": step}
    return job


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    extra = []
    if "--" in argv:
        k = argv.index("--")
        argv, extra = argv[:k], argv[k + 1:]

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="steps:\n" + "\n".join(f"  {k:9s} {v['profile'].upper()}  {v['what']}"
                                      for k, v in STEPS.items()))
    ap.add_argument("step", choices=list(STEPS))
    ap.add_argument("--name", help="job name (default fb-<step>-<timestamp>)")
    ap.add_argument("--tasks", type=int, default=1,
                    help="detect/tag: machines-worth of tasks, normally one per clip")
    ap.add_argument("--hours", type=float, help="max run time per task (cost guard)")
    ap.add_argument("--spot", action="store_true", help="use Spot VMs for this job")
    ap.add_argument("--out", help="where to write the JSON (default jobs/<name>.json)")
    a = ap.parse_args(argv)

    name = a.name or f"fb-{a.step}-{time.strftime('%Y%m%d-%H%M%S')}"
    if not NAME_RE.match(name):
        sys.exit(f"invalid job name {name!r}: lowercase letters, digits and '-', "
                 f"starting with a letter, at most 63 characters")

    job = render(a.step, name, a.tasks, extra, a.hours, "SPOT" if a.spot else None)
    out = Path(a.out) if a.out else ROOT / "jobs" / f"{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(job, indent=2) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
