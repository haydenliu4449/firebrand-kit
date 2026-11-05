# Cloud Batch workflow — firebrand-kit (2026-09-27)

> History: how the Batch setup came about and the lab template's quirks, as
> written when it was set up (moved from the claude.ai project on 2026-11-05).
> The "Status" and "Open items" sections are from 2026-09-27 and long done.
> The current workflow is in README.md and CLAUDE.md.

Replaces the Workbench notebook VM for all compute. Based on the lab template
https://github.com/berkeley-firelab/gcp-example (Daniel San Martin). Full usage
is in the kit's README ("Running on Google Cloud: Batch jobs").

## Why
A Workbench VM bills every hour it is on, idle or not, and its disk bills while
stopped. A Batch job creates a VM, runs one container command, writes results to
the bucket and deletes the VM. You pay only for the minutes the job runs.

## How the lab template works
`.env` (project, bucket, region, registry, image) → `01` build image + push to
Artifact Registry → `02` render `job.json` from a template → `03` upload inputs
to the bucket → `04` `gcloud batch jobs submit` → Batch VM (g2-standard-4 + L4,
`installGpuDrivers`) runs the container with the bucket mounted by GCS FUSE at
`/mnt/disks/share` → `05` rsync outputs back.

## What the kit adds on top
- Steps (one job each): `smoke` (GPU, ~5 min: torch + GPU + resume check),
  `detect` (CPU, one task per clip), [review contact sheets → reject.txt],
  `train` (GPU: tiles → train → eval), `tag` (GPU: model tags every clip →
  `work/tags/<clip>/`). `tiles` also available as a CPU step.
- `scripts/00-setup.sh` (APIs, bucket, registry, IAM for the default compute SA,
  GPU quota check) and `06-jobs.sh` (list / running / status / logs / cancel).
- CPU machine (n2-standard-4, ~$0.19/h) for detect; GPU (g2-standard-4, ~$0.71/h)
  only for smoke/train/tag.
- Everything writes straight into the bucket, and each stage marks itself done
  only when it finishes. Training saves `last.pt` every epoch and resumes from it.
  Jobs retry only on Spot preemption (exit 50001), so `PROVISIONING=SPOT` is safe.
- `maxRunDuration` on every step as a cost guard; job labels app=firebrand, step=<step>.
- Settings in `.env` (copy from `.env.example`; the file could not be written
  remotely).

## Evaluating tags on real footage (added 2026-09-27)
`evaluate_tags.py` (runs locally or in Cloud Shell after `05-gcs_sync.sh review`
and `tags`) compares the model's tracks with the verified classical tracks
(accepted minus `work/clips/<clip>/reject.txt`). Match rule: ≥50% of a track's
detections within 5 px, frame by frame. Model tracks → confirmed / known false
(matches a rejected classical track) / model-only. Precision is a range until the
model-only contact sheets (`work/eval/<clip>/model_only/`) are reviewed into
`work/tags/<clip>/reject.txt`; then it is exact, and the rest count as new finds.
Also reports recall vs verified, fragmentation, ballistic residual, sheets of
missed verified tracks, and a colour overlay (green both / orange model-only /
red missed). Only frames both runs covered are compared. Checked on a
known-answer case (identical tracks + 2 extras): recall 1.00, 5 known-false,
precision 0.96–0.97 → 0.97 after review.
Other levels: synthetic `report.json` (05 results) and `ballistic_rms_px`.

## Notes on the lab template
- **Put the full command in the job.** First smoke job (fb-smoke-20260928-003520)
  passed only `commands: ["smoke"]` and relied on the image ENTRYPOINT; Batch
  ran "smoke" as the program → exit 127 in 0.4 s. The template's approach
  (full command in `commands`) is correct. The earlier note here claiming the
  template double-ran its script was WRONG — retracted. The kit now sets
  `entrypoint: /opt/venv/bin/python` and `commands: [/app/batch_entry.py, ...]`
  explicitly (commit "explicit entrypoint", 2026-09-27 17:41). No image rebuild
  was needed; only job rendering changed.
- `IMAGE_NAME`/`IMAGE_TAG` in `.env` are never substituted (template hard-codes
  `gcp-example:latest`).
- BUCKET_PATH validation rejects `bucket/folder`, though Batch accepts it.
- buildx builds arm64 too, which is wasted work because Batch VMs are x86.
- Exit code 127 = container could not find its program (job-definition problem).
  Batch's own messages are in log `batch_agent_logs`, not `batch_task_logs`.

## Code changes (2026-09-27)
- `run_all.py`: `--stage` accepts comma lists; `--shard I/N|auto` (reads
  BATCH_TASK_INDEX/COUNT); new `tag` stage; train skips only on `done.json`
  (before, `best.pt` after epoch 1 made an interrupted run look finished).
- `firebrand/train.py`: `last.pt` resume with dataset fingerprint; atomic saves;
  `/dev/shm`-aware DataLoader workers (Docker defaults to 64 MB); per-worker
  augmentation RNG (before, with workers>0 every epoch replayed the same augmentations).
- New: `Dockerfile` (nvidia/cuda:12.8.2-base-ubuntu24.04 + torch 2.8.0),
  `batch_entry.py`, `templates/job.{gpu,cpu}.template.json`, `scripts/00-06`,
  `scripts/render_job.py`, `.gcloudignore`, `.dockerignore`, `.gitattributes`,
  `evaluate_tags.py`.
- Fixed: README merge-conflict markers; `.gitignore` no longer ignores `*.json`
  (it was hiding masks/*.json and would have hidden the job templates).

## Status (2026-09-27)
- Project berkeley-fire-lab, bucket gs://berkeley-fire-lab-firebrands, us-central1.
- Image built and pushed; masks and clips uploaded to the bucket.
- First smoke job failed with exit 127 (entrypoint issue above). The fix is on the
  laptop; it needs push → `git pull` in Cloud Shell → resubmit smoke.
- New GitHub repo created from the kit with fresh history (clips no longer in git).

## Verified vs not
- Verified locally: detect (sharded across 2 tasks) and tiles on clip5 through
  `batch_entry.py` with a fake bucket; tag stage and train skip/resume/force logic
  with a stand-in model; all scripts run against a fake gcloud; rendered job JSON
  checked; evaluate_tags on a known-answer case.
- NOT verified yet: real torch training/inference/resume on Batch — the `smoke`
  job tests exactly these.

## Open items
- Rerun smoke with the entrypoint fix; then detect.
- Check L4 GPU quota in the region before GPU jobs.
- Delete the old Workbench instance if one exists (disk bills while stopped).
