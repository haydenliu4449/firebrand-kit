#!/usr/bin/env python3
"""What the container runs on Cloud Batch. You never call this by hand there --
scripts/04-submit_job.sh puts the right arguments in the job.

    batch_entry.py smoke               prove the GPU + torch path works (~5 min)
    batch_entry.py run --stage detect  run_all.py against the mounted bucket
    batch_entry.py hardware            just record what machine this is

The bucket (BUCKET_PATH in .env) is mounted by Batch at /mnt/disks/share, so
inside the container the layout is:

    /mnt/disks/share/clips/        your footage          (scripts/03 uploads it)
    /mnt/disks/share/masks/        mask JSON files       (scripts/03 uploads it)
    /mnt/disks/share/work/         everything produced   (scripts/05 pulls it back)
    /mnt/disks/share/work/logs/<job>/task<N>.log   full log of every task

Everything is written straight into the bucket as it is produced, which is what
makes a job resumable: a stage that finished is in the bucket and the next job
skips it, whatever happened to the machine that ran it.

To try the exact same thing on your own machine, point FB_SHARE at a folder
laid out like the bucket:

    FB_SHARE=./bucket python batch_entry.py run --stage detect --max-frames 60

Why the arguments are passed this way: the image's ENTRYPOINT is this script,
and Batch APPENDS the job's `commands` to the ENTRYPOINT rather than replacing
it. The lab's gcp-example sets an ENTRYPOINT *and* repeats the program in
`commands`, so its job actually runs `hardware_info.py python hardware_info.py
--input ...` -- keep `commands` to arguments only.
"""
from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

APP = Path(__file__).resolve().parent
SHARE = Path(os.environ.get("FB_SHARE", "/mnt/disks/share"))
WORK = SHARE / "work"


def job_id():
    """A stable name for this job's log folder."""
    j = os.environ.get("FB_JOB_NAME") or os.environ.get("BATCH_JOB_ID")
    return j or time.strftime("local-%Y%m%d-%H%M%S")


def task_index():
    return os.environ.get("BATCH_TASK_INDEX", "0")


# ---------------------------------------------------------------------------
# hardware record -- the lab template's hardware_info.py, folded in
# ---------------------------------------------------------------------------

def _run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def hardware_info():
    info = {"platform": platform.platform(), "python": sys.version.split()[0],
            "cpu_count": os.cpu_count()}
    try:
        mem = Path("/proc/meminfo").read_text()
        kb = int(re.search(r"MemTotal:\s*(\d+)", mem).group(1))
        info["memory_gb"] = round(kb / 1024 / 1024, 1)
        cpu = Path("/proc/cpuinfo").read_text()
        m = re.search(r"model name\s*:\s*(.+)", cpu)
        info["cpu_model"] = m.group(1) if m else None
    except (OSError, AttributeError):
        pass
    try:
        info["shm_gb"] = round(shutil.disk_usage("/dev/shm").total / 1e9, 2)
    except OSError:
        info["shm_gb"] = None
    smi = shutil.which("nvidia-smi")
    info["nvidia_smi"] = _run([smi, "--query-gpu=name,driver_version,memory.total",
                               "--format=csv,noheader"]) if smi else None
    try:
        import torch
        info["torch"] = torch.__version__
        info["torch_cuda"] = torch.version.cuda
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception as e:  # noqa: BLE001
        info["torch"] = f"import failed: {e}"
    for k in ("BATCH_JOB_ID", "BATCH_TASK_INDEX", "BATCH_TASK_COUNT", "FB_JOB_NAME"):
        if os.environ.get(k):
            info[k] = os.environ[k]
    return info


# ---------------------------------------------------------------------------

def build_command(mode, rest):
    py = sys.executable
    if mode == "smoke":
        return [py, "-u", str(APP / "tests" / "smoke_train.py")]
    if mode == "run":
        cmd = [py, "-u", str(APP / "run_all.py"),
               "--clips", str(SHARE / "clips"), "--work", str(WORK),
               "--shard", "auto"]
        if "--mask" not in rest and "--no-mask" not in rest:
            name = os.environ.get("FB_MASK", "left_front.json")
            mask = SHARE / "masks" / name
            if mask.exists():
                cmd += ["--mask", str(mask)]
            else:
                print(f"!! no mask at {mask} -- only the burned-in overlays will be "
                      f"excluded. Upload one with scripts/03-upload_inputs.sh.",
                      flush=True)
        return cmd + rest
    return None


def main(argv):
    mode = argv[0] if argv else "smoke"
    rest = argv[1:]
    if mode not in ("smoke", "run", "hardware"):
        print(__doc__)
        return 2

    job, task = job_id(), task_index()
    log_dir = WORK / "logs" / job
    local_log = Path(tempfile.gettempdir()) / f"{job}-task{task}.log"

    hw = hardware_info()
    print(f"firebrand batch task {task} of job {job}", flush=True)
    print(json.dumps(hw, indent=2), flush=True)
    if not SHARE.exists():
        print(f"!! {SHARE} does not exist. On Batch that means the bucket volume "
              f"is missing from the job JSON; locally, set FB_SHARE.", flush=True)
        return 1

    if mode == "hardware":
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / f"task{task}-hardware.json").write_text(json.dumps(hw, indent=2))
        return 0

    cmd = build_command(mode, rest)
    print("$ " + " ".join(cmd), flush=True)

    # Tee the child's output: stdout goes to Cloud Logging, the local copy goes
    # into the bucket at the end. It is written locally first because the
    # bucket mount only uploads a file when it is closed -- appending to a log
    # in the bucket line by line would re-upload the whole file every flush.
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    t0 = time.time()
    rc = 1
    proc = None

    # This script is PID 1 in the container, and PID 1 ignores SIGTERM unless
    # it installs a handler -- so a cancelled or preempted job would sit until
    # SIGKILL and never save its log. Forward the signal to the pipeline and
    # fall through to the log copy below.
    def _stop(signum, _frame):
        if proc is not None and proc.poll() is None:
            proc.send_signal(signum)
        raise KeyboardInterrupt

    import signal
    signal.signal(signal.SIGTERM, _stop)

    with open(local_log, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(hw, indent=2) + "\n$ " + " ".join(cmd) + "\n")
        try:
            proc = subprocess.Popen(cmd, cwd=str(APP), env=env, text=True,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    bufsize=1)
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                fh.write(line)
            rc = proc.wait()
        except KeyboardInterrupt:
            if proc is not None and proc.poll() is None:
                proc.terminate()
            rc = 143
            fh.write("\nstopped by signal (cancelled or preempted)\n")
        except OSError as e:
            fh.write(f"\ncould not start the pipeline: {e}\n")
            print(f"!! could not start the pipeline: {e}", flush=True)
            rc = 1
        finally:
            msg = f"\nexit code {rc} after {(time.time() - t0) / 60:.1f} min\n"
            fh.write(msg)
            sys.stdout.write(msg)

    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local_log, log_dir / f"task{task}.log")
        (log_dir / f"task{task}-hardware.json").write_text(json.dumps(hw, indent=2))
        print(f"log saved to {log_dir / f'task{task}.log'}", flush=True)
    except OSError as e:
        print(f"!! could not save the log to the bucket: {e}", flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
