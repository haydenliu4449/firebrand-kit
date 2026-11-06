# firebrand-kit

UC Berkeley Fire Research Lab. Footage of the 2025 Los Angeles fires from one
house-mounted camera ("Left Front"; 3840×2160, 24 fps, HEVC). Goal: tag
firebrands (embers) and recover their trajectories, above all **saltation**,
embers hopping along the ground. Explain cloud and ML steps plainly, and say
what a job will cost before submitting it.

## How it works
A firebrand is defined by its motion, not its look: a 7 px ember and a 7 px
paver joint are identical in one frame. So:
1. A classical track-before-detect pipeline (stabilise → blue channel →
   top-hat → rolling-median residual → `minAreaRect` streaks → linker → track
   acceptance) gives verified auto-labels (`firebrand/detect.py`, `track.py`).
2. A small U-Net (`firebrand/model.py`; 3-frame residual stack in, 2 px streak
   mask out) trains on synthetic embers painted onto real footage, real
   firebrand-free footage (`negatives.py`) and, with `--real-positives`, real
   verified/tagged firebrands (`reallabels.py`).
3. The model tags every clip, through the same tracker.

Every finding, measurement and decision so far, with the reasons:
`docs/notes/approach.md`. Read the relevant finding before changing
detection, tracking, negatives or training.

## Where things run
- Code is here (github.com/haydenliu4449/firebrand-kit). Clips and results are
  not in git; `work/` is a partial local mirror of the bucket (`05-gcs_sync.sh`).
- Compute is Google Cloud Batch only, through `scripts/00`–`06`: project
  `berkeley-fire-lab`, bucket `gs://berkeley-fire-lab-firebrands`, us-central1;
  settings in `.env` (copied from `.env.example`). There is no GPU or torch
  locally: never train or tag on this machine.
- **A job runs the container image, not this checkout.** After changing
  anything a job uses (`firebrand/`, `run_all.py`, `batch_entry.py`,
  `Dockerfile`, `requirements-batch.txt`): tests → commit → push →
  `bash scripts/01-build_and_push.sh` → then submit. On 2026-10-22 a job
  failed because the image was older than the code. The command line at the
  top of a job's log shows which flags the image knows (e.g. `--neg-clips`).
- In Cloud Shell: `$HOME` is 5 GB and the VM stops after ~40 min idle (jobs
  keep running; `claude -c` resumes this session). Don't pull overlay videos
  in bulk.
- Cloud Shell restarts often and wipes `/tmp`, including Claude's scratchpad. Any
  local run over a few minutes writes results as it goes to the git-ignored
  `work/` and skips finished work on rerun (`work/recall_review/` is the example).

## Commands
| what | command |
|---|---|
| unit tests (seconds, no torch) | `python3 tests/test_negatives.py && python3 tests/test_reallabels.py && python3 tests/test_detect.py && python3 tests/test_resting.py` |
| tracker benchmark (~10 s) | `python3 tests/benchmark.py` (`--ablate` for ablations) |
| submit a step | `bash scripts/04-submit_job.sh <smoke\|detect\|tiles\|train\|tag> [-- run_all.py args]` |
| job state / logs | `bash scripts/06-jobs.sh [list\|running\|status JOB\|logs JOB\|log JOB N]` |
| pull results | `bash scripts/05-gcs_sync.sh <review\|results\|tags\|logs>` (`tags --video` = big overlays) |
| upload clips, negclips, masks, negatives.txt, reject.txt | `bash scripts/03-upload_inputs.sh` |
| score tags against verified tracks | `python3 evaluate_tags.py [--heldout] [--no-sheets]` |

Torch code (`train.py`, `infer.py`, `model.py`) can only run on Batch; the
`smoke` job (~5 min, ~$0.10) exercises it. If `evaluate_tags.py` lacks
modules: `pip3 install --user opencv-python-headless scipy pandas`.

## Cost, and what to ask before doing
- GPU (smoke/train/tag): g2-standard-4 + L4 ≈ $0.71/h; a train with negatives
  and real positives ≈ 1–2 h. CPU (detect/tiles): n2-standard-4 ≈ $0.19/h.
  Bucket downloads ≈ $0.12/GB.
- Ask first: submitting any job, rebuilding the image, uploading over bucket
  files, cancelling or deleting jobs, deleting anything in the bucket.
  Checking jobs, reading logs and pulling small results need no asking.

## Rules learned the hard way
- `--force`: a stage with `done.json` is skipped without it. A failed `train`
  resubmitted **without** `--force` resumes from `last.pt`; with it, restarts.
- Re-running detect or tag renumbers tracks. The stage moves the bucket's old
  `reject.txt` aside; delete local `work/tags` (and stale local
  `work/clips/*/reject.txt`) before pulling, or 03 re-uploads stale reviews.
- Held-out frames: every 4th 32-frame block of each clip
  (`negatives.is_heldout`) is never trained on, for negatives and real
  positives alike; `evaluate_tags.py --heldout` scores exactly those. Changing
  `BLOCK`/`EVAL_EVERY` breaks every before/after comparison.
- Long firebrand-free videos go in `negclips/`, never `clips/`: detect and tag
  process all of `clips/`, at 2–4 s CPU per 4K frame.
- `work/negatives.txt`: `<clip-stem> <a-b frames | m:ss-m:ss times>, ...`.
  Never list a frame with even one faint real ember; it teaches faint embers
  (the ones on the ground, i.e. saltation) as background.
- CSVs from detect/tag always get a header; read them with `read_csv_or_empty`
  (a clip with no tracks once wrote a bare newline that crashed training).
- After touching the linker or detector, `tests/benchmark.py`'s "mirrored"
  row must still equal "shipped defaults" (the linker once couldn't follow
  embers flying left: 0 tracks mirrored).
- Contact sheets can't show a track hopping between neighbouring embers. Before
  trusting a trajectory from a dense shower, draw a path view (max of the
  track's frames with its boxes).
- Pixel limits scale with width/1920 (`run_all.px_scale`); never hard-code
  pixel sizes for 4K.
- Real-firebrand training removes unlabelled embers from the labelled frame
  per pixel, not per blob (bright glows merge in dense showers). Keep it so.
- Say what was verified and how. This machine can't run torch, so model-side
  changes are "plumbing tested, effect unmeasured" until a Batch run shows it.
- Keep `README.md` (the user-facing how-to) in step with any workflow change.

## Layout
- `run_all.py`: every stage (detect, tiles, train, eval, tag) and their flags
- `batch_entry.py`: container entry on Batch; `scripts/`: 00 setup … 06 jobs
- `firebrand/`: `frames` (decode, IR check), `detect`, `track`, `synth`,
  `dataset`, `model`, `train`, `infer`, `evaluate`, `review` (contact
  sheets), `negatives`, `reallabels`, `resting`, `gcsio`
- Local tools: `evaluate_tags.py`, `suggest_negatives.py`, `make_mask.py`,
  `run_pipeline.py` (one clip through detect/track, e.g. to check a mask)
- `tests/`: unit tests, `benchmark.py`, `smoke_train.py` (the smoke job)
- Legacy, don't use: `cloud/` (old Workbench VM scripts), `notebooks/`

## Last known state (2026-11-06)
- Tagged 2026-11-06 with the 10-23 model and mask `left_front_v2.json`: held-out
  recall 0.005 → 0.443 (0.509 without canopy tracks; finding 9). Old tags kept in
  `work/tags_before_20261023`. Next: keep canopy tracks out of the labels, then
  `train -- --force --real-positives` (first train with negatives) → tag → compare.
- Negatives: 30-min firebrand-free video `D01_20250107230423` in `negclips/`
  (3 stretches, ~26 s) in `work/negatives.txt`.
- Open: `--real-positives` retrain not reported; track identity in dense showers
  (finding 5); chance alignments in dense showers, up to ~10-15% of clip5's
  tracks (finding 9); joining descent+ascent pieces per hop; exposure time, true
  capture rate, IR switching.
