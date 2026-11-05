---
name: check-job
description: Check a Cloud Batch job (or the most recent ones) and summarise its state, progress and the log lines that matter. Use when asked whether training or tagging finished, how a job went, or why it failed.
argument-hint: "[job-name]"
---

Check the Batch job `$ARGUMENTS` (if empty, the most recent jobs). Read-only:
never cancel, delete or resubmit from this skill.

1. `bash scripts/06-jobs.sh list` to find the job if no name was given, then
   `bash scripts/06-jobs.sh status <job>`.
2. RUNNING: `bash scripts/06-jobs.sh logs <job>` and report where it is
   (stage, epoch N of M, clip k of n) and roughly how long is left. Jobs keep
   running when this session or Cloud Shell closes.
3. SUCCEEDED or FAILED: `bash scripts/06-jobs.sh log <job> 0` (other task
   numbers for per-clip steps). Report, as applicable:
   - train: the `negatives.txt:` line, blocks dropped by the check (`!!` lines
     with video times), `hard negatives:` / `real firebrands:` lines, epochs and
     best val F1, and the eval table of false tracks per 100 firebrand-free
     frames with the threshold it chose.
   - tag: tracks per clip, and any clip with 0 tracks (say whether that clip
     should have firebrands; if so the threshold may be too cautious).
   - detect: accepted/rejected tracks per clip.
   - FAILED: the traceback, its cause in this repo's code, and the fix. Check
     whether the job's command line shows the flags the current code expects;
     if not, the image is stale and needs `bash scripts/01-build_and_push.sh`.
4. For a finished train, `gcloud storage cat
   gs://berkeley-fire-lab-firebrands/work/runs/streak/done.json` confirms every
   epoch ran.
5. End with the next step (e.g. tag after train, review after detect), without
   running it.
