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
| unit tests (seconds, no torch) | `python3 tests/test_negatives.py && python3 tests/test_reallabels.py && python3 tests/test_detect.py && python3 tests/test_resting.py && python3 tests/test_tiles.py && python3 tests/test_track.py && python3 tests/test_trajectory.py && python3 tests/test_calibration.py && python3 tests/test_tag_tool.py && python3 tests/test_pathoverlay.py` |
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
- Exit code -9 in a job log means killed for memory (15.6 GB VM), with no
  traceback. Tile masks are uint8 in the tiles stage: use `dataset.to_mask8`, never
  `.astype`, which zeroes a 0..1 mask.
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
  sheets), `negatives`, `reallabels`, `resting`, `trajectory`, `calibration`
  (pixels -> metres, `calib/left_front.json`), `pathoverlay` (whole paths drawn on
  the video), `gcsio`
- Local tools: `evaluate_tags.py`, `trajectories.py` (tracks -> trajectories on existing
  tags, path views), `path_overlay.py` (overlay video with every firebrand's whole path,
  from existing tags; `--overlay-paths` is the same on a detect/tag job), `tag_tool.py` (hand tagging, runs on the user's laptop: rotated
  boxes fitted to the ember, CSV in the kit's columns), `suggest_negatives.py`, `make_mask.py`,
  `run_pipeline.py` (one clip through detect/track, e.g. to check a mask)
- `tests/`: unit tests, `benchmark.py`, `smoke_train.py` (the smoke job)
- Legacy, don't use: `cloud/` (old Workbench VM scripts), `notebooks/`

## Last known state (2026-11-20)
- Current tags (bucket `work/tags`): the **11-07 model** set, threshold 0.8, job
  fb-tag-20261107-222905, mask `left_front_v3.json`, tag tracker of finding 10, with
  trajectories made locally by `trajectories.py` (finding 11). Box coverage of verified
  held-out tracks 66.3%; 1,352 hop trajectories, 256 with crowding <= 2. 10,311 objects.
  These were RESTORED on 2026-11-20 from `work/tags_20261120` after the 11-20 retrain's
  tags scored worse at a 5 px match (0.382 -> 0.355), level at 15 px (0.582 -> 0.592) and
  13% more numerous (finding 15).
- The 11-20 retrain's tags are archived at bucket `work/tags_retrain_20261120` (10,313
  objects, threshold 0.6, with the job's own trajectories and whole-path overlays). They
  are the evidence behind finding 15; don't use them for science.
  `work/tags_20261120` is the identical copy of the restored good set.
  Earlier tags: `work/tags_20261107`, `work/tags_20261106`, `work/tags_before_20261023`
  (bucket: tags_20261107, tags_20261106). Models: bucket `runs/streak` (11-20),
  `runs/streak_20261107`, `runs/streak_20261023`.
- **A tag set's `done.json` does not identify its model**: it records the path
  `runs/streak/best.pt`, which `--force` overwrites each retrain. The restored 11-07 tags
  therefore claim `runs/streak/best.pt` too. Right now only `threshold` tells them apart
  (0.8 = 11-07 model, 0.6 = 11-20). Worth writing the model's digest or run name instead.
- Locally (Windows checkout, git-ignored): `work/tags_retrain_20261120/` = the 11-20
  tags' CSVs, `work/baseline/` = the restored 11-07 tags scored against the same classical
  CSVs, `work/eval*/summary*.csv` = both scorings, `work/clips/*/` = classical
  detections + the hand `reject.txt` files. There is deliberately no local `work/tags`.
  Only CSVs were pulled, not contact sheets or overlay videos.
- Methodology + change log for the lab: https://claude.ai/artifact/J5UvGqMa53v3gg2M1zWioo
  (keep it in step with approach.md).
- Metres (finding 12): lens + ground plane from the pavers, scale PROVISIONAL (slabs
  assumed 60 x 120 cm; street then 5.2 m wide, suspiciously narrow). Next: a measured
  length from the user → rescale `calib/left_front.json` → rerun `trajectories.py`;
  faint long streaks remain the main miss. L4 VMs can be scarce: queueing is free.
- 2026-11-15: whole-path overlay added (finding 13). `path_overlay.py` works on existing
  tags with no job. Tracking is unfinished by the user's plan: the overlay draws whatever
  IDs exist.
- 2026-11-20: image REBUILT (digest 469a1e19); `--overlay-paths` verified on a real job
  (finding 13), and it writes into `overlay.mp4`, not a separate file.
  `--real-positives` retrain done (finding 14): on synthetic + empty frames it looked
  much better (false tracks per 100 empty frames 16.88 -> 4.55, recall unchanged,
  threshold 0.6). On real footage its tags scored worse at a 5 px match and no better at
  15 px, and were rolled back the same day (finding 15). **Do not trust the synthetic or
  empty-frame eval for field recall; score tags with `evaluate_tags.py` against verified
  tracks before shipping a model.**
- Negatives: 30-min firebrand-free video `D01_20250107230423` in `negclips/`
  (3 stretches, ~26 s) in `work/negatives.txt`.
- Open: whether the 11-20 result is the model or the threshold (needs one tag job at the
  other threshold, ~2 h ~$1.50; the per-detection score is in no CSV, so it cannot be
  re-thresholded offline — writing it would make this free); no box-coverage figure
  exists for the 11-20 tags, and no tool in the repo computes one; track identity in
  dense showers
  (finding 5); chance alignments in dense showers, up to ~10-15% of clip5's
  tracks (finding 9); joining descent+ascent pieces per hop; exposure time, true
  capture rate, IR switching.
