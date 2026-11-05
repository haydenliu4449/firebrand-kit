---
name: run-step
description: Submit one pipeline step (smoke, detect, tiles, train, tag) as a Cloud Batch job, follow it, and summarise the result.
argument-hint: "<step> [-- run_all.py args]"
disable-model-invocation: true
---

Run the pipeline step `$ARGUMENTS` on Cloud Batch.

1. **Is the image current?** If code a job uses has changed since the last
   `scripts/01-build_and_push.sh` (uncommitted edits, or commits since), stop
   and offer `/deploy` first. A job runs the image, not this checkout.
2. **Say what it will do and cost**, then confirm the flags:
   - train retrains only with `-- --force` (else it is skipped as done; a
     failed train resubmitted without `--force` resumes). Add
     `--real-positives` to train on real tagged firebrands too.
   - tag usually wants `-- --force --overlay`.
   - detect on new clips: `-- --overlay`; re-running it renumbers tracks.
   - GPU (smoke/train/tag) ≈ $0.71/h; CPU (detect/tiles) ≈ $0.19/h.
3. Submit: `bash scripts/04-submit_job.sh $ARGUMENTS`. Note the job name.
4. Follow it with `bash scripts/06-jobs.sh status <job>` every few minutes,
   not continuously. Train and tag take hours; if the session has to end, say
   so and tell the user to run `/check-job <job>` later.
5. When it ends, summarise it as `/check-job` does: key log lines, failures
   and their cause, and the next step.
