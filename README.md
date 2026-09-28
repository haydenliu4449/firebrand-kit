# firebrand-kit

Detecting and tracking firebrands in single-view security footage, for
saltation trajectory measurement.

Built for the UC Berkeley Fire Research Lab firebrand project (Ranch Top Rd,
Pasadena, 2025-01-07 Eaton Fire footage). Single camera, no overlapping views.

---

## The one idea

**A firebrand is defined by its motion, not its appearance.**

In a single frame, a 7-pixel glowing ember and a 7-pixel sunlit crack in the
concrete are the same object: a small, bright, elongated thing. No appearance
feature separates them, so no per-frame threshold can, and a detector trained
on per-frame thresholds inherits every mistake the threshold made.

So this pipeline runs a deliberately **over-sensitive** per-frame detector and
lets **temporal coherence** decide what was real. A candidate becomes a
firebrand only if it joins a track of three or more frames with a smooth
heading and consistent brightness. Static texture never does; embers always do.

That inversion — the tracker is the detector — is what makes the auto-labels
trustworthy enough to train on, and it is the difference from the first
attempt, where OpenCV thresholds generated labels that YOLO then learned to
imitate, mistakes included.

---

## Measured results

Benchmark: 50 frames of synthetic firebrands composited additively (in linear
light) onto a real frame from your camera, with 6% brightness flicker and 0.8 px
of camera shake. Ground truth exact by construction.

```
                              raw detector          after tracking       tracks
shipped defaults          P=1.00  R=0.75         P=1.00  R=0.45      recall 0.62, frag 1.12, 0 ID switches
```

Reproduce with `python tests/benchmark.py --ablate --channels`.

### What each piece is worth

```
  no stabilisation          P=0.08     (2877 detections, 12x too many)
  no edge suppression       P=0.17     (1392 detections)
  neither                   P=0.06     (4128 detections)
  shipped defaults          P=1.00     ( 237 detections)
```

**Stabilisation is the single most important step after the channel choice, and
the easiest to leave out.** These cameras are mounted on houses during a wind
event. A *sub-pixel* tremor is enough to break rolling-median background
subtraction: when a high-contrast edge moves half a pixel, the median stops
cancelling it and it reappears in the residual as a bright, elongated,
streak-shaped object — indistinguishable from a firebrand by shape, and present
in every frame. Phase-correlation alignment plus a gradient-proportional
threshold surcharge (`edge_suppress`) removes them.

### Where the blue channel actually matters

The background is fire-glow orange (mean RGB 154, 87, 23); firebrands are
near-blackbody white (250, 241, 227). Single-frame separability, measured on
your frame, in background sigmas:

```
  red 3.6  |  grayscale 9.0  |  green 9.1  |  blue 19.7  |  blue-0.15*red 22.1
```

But most of that advantage is redundant once background subtraction is working.
Measured, rather than assumed:

```
  ember brightness      blue recall   gray recall      blue trkR   gray trkR
  bright (70-255)          0.73          0.72             0.42        0.42
  medium (35-90)           0.80          0.51             0.60        0.40
  faint  (20-55)           0.71          0.19             0.64        0.00
```

Blue and grayscale are equivalent on bright embers. Blue is the difference
between working and not working on faint ones — which is exactly the population
that matters here, because an ember is dimmest while it is on the ground losing
heat, and **ground contact is the saltation event you are trying to measure.**

---

## Running on Google Cloud: Batch jobs, not a notebook VM

The compute runs as **Cloud Batch jobs**, using the lab's workflow from
[berkeley-firelab/gcp-example](https://github.com/berkeley-firelab/gcp-example).
Each job starts a VM, runs one command in a container, writes its results
into the bucket, and deletes the VM. You pay for the minutes the job runs.
There is nothing left switched on afterwards, so nothing to forget.

That is the whole cost difference from the Workbench notebook this kit used
before: a notebook VM bills every hour it is on, whether or not a cell is
running, and its disk bills even when it is stopped.

### How the workflow fits together

```
 on your side (Cloud Shell, or a laptop with gcloud)        Google Cloud
 ─────────────────────────────────────────────              ─────────────────────────────────────
 01-build_and_push  Dockerfile + code  ──► Cloud Build ──►  Artifact Registry: the image
 03-upload_inputs   clips/, masks/, reject.txt  ───────────►  bucket  gs://BUCKET_PATH/
 04-submit_job STEP jobs/<name>.json ──► Batch ──► VM (+GPU): pulls the image, mounts the
                                                    bucket at /mnt/disks/share, runs
                                                    run_all.py, writes work/ into the
                                                    bucket, VM deleted when it ends
 05-gcs_sync        work/ (sheets, CSVs, model, tags) ◄────  bucket
 06-jobs            status / logs / cancel
```

Those are the lab template's five numbered scripts, in the same order,
adapted to this pipeline, plus `00-setup.sh` (one-time) and `06-jobs.sh`
(watching jobs). All settings are in `.env`.

| file | what it is |
|---|---|
| `.env` (copy of `.env.example`) | project, bucket, region, image name, machine types |
| `Dockerfile` | the image every job runs: CUDA base, torch, OpenCV, ffmpeg, this kit |
| `batch_entry.py` | the container's entrypoint: maps the bucket paths, runs `run_all.py`, saves the log |
| `templates/job.gpu.template.json`, `job.cpu.template.json` | Batch job templates (GPU for smoke/train/tag, CPU for detect/tiles) |
| `scripts/render_job.py` | fills a template for one step; the list of steps lives here |
| `jobs/` | every job file that was submitted, kept as a record of what ran |

### The steps

Each step is one job. Run them in this order:

| step | machine | what it does | then |
|---|---|---|---|
| `smoke` | GPU | builds toy tiles, trains 2 epochs, **interrupts and resumes a run**, runs inference on the GPU. ~5 min | check that it passed |
| `detect` | CPU | the classical detector on every clip, one task per clip, several machines in parallel | **review the contact sheets** |
| `train` | GPU | tiles (skipped if built) → train → eval against the classical detector | read `report.json` |
| `tag` | GPU | the trained model tags every clip → `work/tags/<clip>/` | the output you use |

(`tiles` also exists as a CPU step if you want to build the tiles without paying for a GPU.)

Every job pays a fixed overhead of a few minutes to boot the VM and pull the
~6 GB image. Rough prices, on-demand in us-central1: GPU machine
(`g2-standard-4` + L4) about $0.71/hour, CPU machine (`n2-standard-4`) about
$0.19/hour. The kit's own estimate for 30 epochs on ~8k tiles is 15–25 minutes
on a T4, and an L4 should be no slower, so a full train step is on the order of
a dollar. Set `PROVISIONING=SPOT` in `.env` to pay much less, with the risk
that a machine is taken back mid-job (see Resuming below).

### First time

Run the scripts in **Cloud Shell** (the `>_` icon in the Cloud Console): `gcloud`
and Docker are preinstalled and already logged in, and bash scripts just work.
From Windows, Git Bash with the Google Cloud SDK installed also works.

```bash
git clone https://github.com/haydenliu4449/Firebrands firebrand-kit && cd firebrand-kit
cp .env.example .env && nano .env      # check PROJECT_ID, BUCKET_PATH, LOCATION

bash scripts/00-setup.sh               # APIs, bucket, registry, permissions, GPU quota check
bash scripts/01-build_and_push.sh      # ~10 min; again after any code change
bash scripts/03-upload_inputs.sh       # clips/ and masks/ -> the bucket
bash scripts/04-submit_job.sh smoke    # prove the GPU path before anything real
bash scripts/06-jobs.sh status fb-smoke-...   # the name 04 printed
```

`00-setup.sh` checks your GPU quota. A new project often has a GPU quota of
**zero**, and then a GPU job does not fail. It sits in `SCHEDULED` waiting for
quota. Request an increase (IAM & Admin → Quotas, `NVIDIA_L4_GPUS` in your
region) before the day you need it.

### The normal loop

```bash
bash scripts/04-submit_job.sh detect -- --max-frames 300 --overlay   # start small
bash scripts/05-gcs_sync.sh review     # contact sheets -> work/clips/<clip>/review/
#   write the IDs that are NOT firebrands in work/clips/<clip>/reject.txt
bash scripts/03-upload_inputs.sh       # pushes the reject.txt files
bash scripts/04-submit_job.sh train    # tiles -> train -> eval
bash scripts/05-gcs_sync.sh results    # prints the classical vs learned comparison
bash scripts/04-submit_job.sh tag      # the model tags every clip
bash scripts/05-gcs_sync.sh tags       # work/tags/<clip>/tracks.csv, detections.csv, labels/
```

Anything after `--` is passed to `run_all.py` unchanged: `--epochs 40`,
`--synth-frames 600`, `--threshold 0.4`, `--overlay`, `--force` to redo a stage
that already finished. None of these need an image rebuild. Only code changes
do.

You can also look at the results without downloading anything: open the bucket
in the Cloud Console and click a contact-sheet PNG.

### Where things live in the bucket

```
gs://BUCKET_PATH/
  clips/                        your footage                    (03 uploads)
  masks/left_front.json         the mask named by MASK in .env  (03 uploads)
  work/clips/<clip>/            detect: tracks.csv, review/, audit.json, reject.txt
  work/data/tiles.npz           training tiles
  work/runs/streak/             best.pt, last.pt, history.json, report.json
  work/tags/<clip>/             tag: tracks.csv, detections.csv, labels/, overlay.mp4
  work/logs/<job>/task<N>.log   the full output of every task
```

### Resuming, retries and Spot

Every stage writes straight into the bucket and marks itself done only at the
end, so re-submitting a step skips whatever already finished. Training goes
further: it saves `last.pt` every epoch and a new run continues from the last
finished epoch. The job templates retry a task automatically, up to 3 times,
when the failure was a Spot preemption (Batch exit code 50001), and not for
real errors. So `PROVISIONING=SPOT` is safe: an interrupted job costs you the
epoch it was in, not the run.

### Cost guards

- **Nothing runs between jobs.** `bash scripts/06-jobs.sh running` lists
  anything queued or running, and says so plainly when nothing is billing.
- **Every step has a time limit** (`maxRunDuration`: 45 min for smoke, 4–6 h for
  the others; override with `--hours`). A hung job is killed rather than billing
  all weekend.
- **Stop a job** with `bash scripts/06-jobs.sh cancel JOB`. Finished stages stay.
- **Retire the old Workbench VM** if you created one: a stopped Workbench
  instance still bills about $15/month for its disk. Delete it with
  `bash cloud/workbench.sh delete` once anything you need from it is in the bucket.
- Budget alerts email you but stop nothing. `cloud/setup.sh` can create one.

### Running a job's exact command on your own machine

`batch_entry.py` reads the bucket through a folder, so any folder laid out like
the bucket works. This is a good way to debug a step without paying for a VM:

```bash
mkdir -p bucket/clips bucket/masks && cp clips/clip5.mp4 bucket/clips/ && cp masks/*.json bucket/masks/
FB_SHARE=./bucket python batch_entry.py run --stage detect --max-frames 60
```

### Two things in the lab template worth knowing

These matter if you reuse gcp-example for something else. The kit already
handles both.

- **The job's `commands` are appended to the image's `ENTRYPOINT`, not
  substituted for it.** The template sets `ENTRYPOINT python hardware_info.py`
  *and* repeats `python hardware_info.py --input ...` in `commands`, so the
  container runs the script with its own path as extra arguments. Here the
  entrypoint is `batch_entry.py` and `commands` holds arguments only.
- **`IMAGE_NAME` and `IMAGE_TAG` in `.env` never reach the job.** The template
  hard-codes `gcp-example:latest` and the renderer only substitutes the
  location, project and repository. `render_job.py` fills the whole image URI.

### The old notebook route

`cloud/` (Workbench VM) and `notebooks/` still work, and the notebook is still
the right place to *look* at results. For anything that computes, use the
steps above.

---

## Scripts, not notebooks, for anything slow

A Jupyter kernel holds everything in memory. When it reconnects (and on a long
run it will) you lose every variable, and the only way back is to start over.
Notebooks are the wrong tool for stages that take tens of minutes.

`run_all.py` is the whole pipeline as a resumable script. It is what every
Batch job runs, and it runs the same way on your own machine:

```bash
python run_all.py --clips clips/ --work work --mask masks/left_front.json --stage detect
python run_all.py --clips clips/ --work work --stage tiles,train,eval
python run_all.py --clips clips/ --work work --stage tag
```

Stages: `detect` → `tiles` → `train` → `eval` → `tag`. `--stage` takes one,
a comma list, or `all`. Redo one with `--force`. Between `detect` and `tiles`,
review the contact sheets and list the non-firebrand track IDs in
`work/clips/<clip>/reject.txt`.

**Use the notebook for looking, the script for producing.** Every stage leaves
files the notebook loads in seconds, so plots and diagnostics stay interactive
without holding an hour of compute in a kernel that might drop.

---

## Stage 0 — check the install, before touching your data

```bash
python tests/benchmark.py --ablate --channels
```

Self-contained (uses the bundled sample frame), takes about a minute, and
reproduces every number above. If `shipped defaults` comes back at P = 1.00 with
0 ID switches, the install is good.

---

## Where your clips go

For local runs there is no upload step: everything runs on files already on
your disk. (For Batch, `scripts/03-upload_inputs.sh` copies `clips/` to the
bucket.) Paths in these examples like `clips/eaton_01.mp4` are placeholders for
*your* filenames.

Either make a folder inside the kit and copy clips into it:

```bash
mkdir clips          # then copy your .mp4 files in
ls clips             # confirm what they are actually called
```

or leave them where they are and pass the full path. On Windows in Git Bash,
the C: drive is `/c/`, and **paths containing spaces need quotes**:

```bash
python make_mask.py "/c/Users/Hayden Liu/Downloads/eaton footage/cam1.mp4" \
    --out masks/left_front.json
```

If a path is wrong, the error tells you the working directory, what the folder
actually contains, and lists video files it can find nearby.

Supported: whatever OpenCV can decode — `.mp4`, `.avi`, `.mov`, `.mkv`. Some
DVRs export `.dav` or H.265, which OpenCV often cannot read; convert first and
keep the quality high, since compression is what destroys these objects:

```bash
ffmpeg -i input.dav -c:v libx264 -crf 18 -an clips/cam1.mp4
```

---

## Stage 1 — draw the mask, once per camera angle

```bash
python make_mask.py clips/YOUR_CLIP.mp4 --out masks/left_front.json
```

Click to add points, ENTER to close a polygon, `s` to save. Press `p` for a
max-projection view, which shows where embers actually fly so you can see
whether a polygon is about to throw away the interesting region.

**Exclude glass and painted metal that reflects the sky.** A reflection is a
mirrored duplicate of a real firebrand: it double-counts flux and generates
trajectories that are physically impossible, which quietly poisons every
saltation statistic downstream. This is the one manual step nobody can do for
you — an earlier version of this kit shipped guessed polygons for the Left Front
camera, and rendered over the real frame they covered driveway instead of glass.
The two burned-in overlay boxes (timestamp, camera label) *are* measured and are
pre-loaded.

### What to draw polygons around

For the Ranch Top Rd 'Left Front' camera, in order of how much damage skipping
each one does. A starter mask is in `masks/left_front_starter.json` — treat it
as a first draft, not an answer.

**A. Vegetation — the hedge and lawn strip across the middle.** The important
one, and the least obvious. Wind-blown leaves are *real motion*, so neither
background subtraction nor stabilisation removes them: the median can't cancel
something that genuinely moves, and stabilisation only corrects the camera. A
swaying leaf edge catching the glow reads as a short bright streak, frame after
frame, from a slightly different place each time — which can even survive the
3-frame coherence rule, because the motion is smooth. This is the one confuser
in the scene that defeats every other defence in the pipeline.

**B. The left wall, light fixture, and the metal handrail.** The rail is
specular, permanently bright, and already streak-shaped. It's static, so the
median handles it in principle — but it sits at maximum contrast, which is
exactly where sub-pixel shake leaves the largest residual.

**C. The street beyond the hedge.** Optional. Embers there are far away and
below your resolution limit, and passing headlights are large moving bright
objects. Cutting it also speeds up every run.

**Do not exclude the driveway.** Pavement texture does generate false positives,
but that is what stabilisation, edge suppression and the 3-frame rule are for —
and the ground plane is where saltation contacts happen, so masking it removes
the measurement you are here to make.

### How much should you mask?

There is no target percentage, and aiming for one will mislead you — the
fraction of *pixels* removed says nothing about what you lost, because the
regions worth cutting are rarely the regions where the measurement happens.

Measured on the sample frame, the driveway is **57% of the frame**, and every
mask variant below keeps ~98% of it:

```
mask                        % frame cut   % of driveway kept   % of single-frame candidates cut
overlays only                      2.4%                98.7%                              6.6%
B + overlays                      18.9%                97.8%                             22.4%
B + A + overlays                  28.6%                97.7%                             36.3%
everything (starter mask)         44.2%                97.7%                             78.9%
```

So "44% of the frame" sounds alarming and costs 2% of the ground plane. The
number that moved was candidates, and almost all of that is region C — the
street holds 47% of the single-frame candidates on its own, because that is
where the airborne ember cast is. Those are real embers; they are just not
*saltating*, and they are past the distance where you can resolve a trajectory.

Two numbers to watch instead of a percentage:

- **accepted tracks lost** — the actual cost. Keep it low.
- **ground plane kept** — above ~95%. A mask that clips a corner of the driveway
  biases *which* hops you observe, and a biased sample of hop lengths is worse
  than a smaller unbiased one.

Both are reported by:

```bash
python run_pipeline.py clips/YOUR_CLIP.mp4 --out work/check \
    --mask masks/left_front_starter.json --check-mask
```

Rough guidance for this camera: **~19% (B + overlays) is a sane starting point.**
Add the vegetation (~29%) if the heatmap shows the hedge is hot. Only go to 44%
if you are strictly measuring ground contacts and are content to discard the
airborne population — and note that the shipped starter mask, run through
`--check-mask`, flags itself as removing about a quarter of accepted tracks.

### Better: let the data draw it

Every run writes `fp_heatmap.png` — rejected candidates as a red density map,
accepted tracks as green dots, both on your frame. Run once with `--no-mask`,
open it, and mask where the red is hot and the green is absent. Where a region
has both, leave it in and let the tracker do the work.

That ordering matters. Vegetation is a genuine judgement call: masking the hedge
throws away any ember that flies in front of it, and on a windy night that may
be a lot of them. The heatmap tells you which cost is larger *on your footage*
instead of making you guess from a still frame — which is how I put the
reflection polygons in the wrong place in the first version of this kit.

Mask specs are stored in normalized coordinates, so one mask works across
resolutions (the sample frame is 1917×1079, not 1920×1080 — an exact-size check
would have silently skipped the mask entirely).

---

## Stage 2 — auto-labels, no training

```bash
python run_pipeline.py clips/eaton_01.mp4 --out work/eaton_01 \
    --mask masks/left_front.json --overlay --max-frames 300
```

Start with `--max-frames 300`. Get one clip right before running forty.

Writes to `work/eaton_01/`:

| file | what it is |
|---|---|
| `audit.json` | **read this first** — IR mode, duplicate frames, channel separability |
| `tracks.csv` | one row per accepted track, with a ballistic-fit residual column |
| `detections.csv` | one row per detection: x, y, streak length, orientation, flux |
| `review/sheet_*.png` | contact sheets for verification |
| `labels/*.txt` | YOLO-format boxes, one file per frame |
| `overlay.mp4` | detections drawn on the clip |

**Look at `overlay.mp4` before anything else.** Most bugs in this pipeline are
obvious in one frame and invisible in every metric.

### Pre-flight checks that actually matter

`audit.json` reports three things that will silently corrupt your results:

- **IR night mode.** If the camera flips to monochrome, the blue channel stops
  helping and the pipeline falls back to grayscale (with the recall cost above).
- **Duplicate frames.** Many DVRs advertise 30 fps by repeating frames from a
  15 fps sensor. If the duplicate fraction is near 0.5, your true capture rate
  is half what the container says and **every velocity you measure is wrong by
  2x**.
- **Channel separability** on your actual clips, not on the one frame these
  defaults were tuned on.

### Exposure time

If you know the camera's exposure time, pass `--exposure 0.002`. Then streak
length converts directly to speed: **speed = L / t_exposure**, an instantaneous
velocity per detection from a single frame, before any tracking.

If you don't, `track.estimate_step_ratio` recovers the equivalent ratio from the
data by asking, for each streak, how far along its own direction the next frame's
detections lie. Note the ratio it returns is defined against *measured* streak
length, which includes PSF broadening, so it is self-consistent for prediction
but biased low as an estimate of `(1/fps)/exposure`. Getting the real exposure
from the camera is worth the effort — it is the only thing standing between you
and absolute speeds.

---

## Stage 3 — verification (a few hours of your time, total)

Drawing a box takes ~4 seconds. Saying yes/no to one already drawn takes ~1. And
a track of six detections is **one** decision instead of six — roughly a 25x
multiplier before you write any model code.

1. Open `review/sheet_*.png`. Each row is one track, shown as a strip of crops
   in time order. A real firebrand sits still in the middle of every cell while
   the background slides past; a false positive is static texture in all of them.
2. Note the IDs that are not firebrands.
3. Re-run with `--reject 12,47,51,88`.

Keep `negatives.csv`. Every rejected track is a **labelled hard negative** mined
from your own footage — a paver joint, a compression block, a reflection — and
those are worth far more to the model than random background crops, because they
are exactly the mistakes it would otherwise make.

At 24 tracks per sheet and ~30 s per sheet, 800 tracks is about half an hour.

---

## Stage 4 — synthetic training data

800 verified tracks are not a large training set. They are an excellent
*parameter estimate* for a generator that can produce an unlimited one.

A firebrand is a moving point emitter integrated over the exposure. That is not
an approximation — it is the same forward model, with parameters you sample,
which is why synthetic data works far better here than it usually does.

```python
from firebrand import Config, synth as S, dataset as DS
import pandas as pd

cfg  = Config()
rows = pd.read_csv("work/eaton_01/tracks.csv").to_dict("records")
cfg.synth = S.fit_config_to_tracks(rows, cfg.synth)     # match YOUR distributions

X, Y, frames, gt = DS.build_from_synthetic(background_frame, 400, cfg)
DS.save_tiles("data/tiles.npz", X, Y)
```

Three details decide whether the model transfers:

1. **Real backgrounds.** Composite onto frames from your own camera. Synthetic
   backgrounds lack your pavement texture, which is the exact confuser the model
   must learn to reject.
2. **Additive, in linear light.** A glowing object adds photons; it does not
   alpha-blend. Compositing in gamma space gives the wrong brightness profile
   and the model learns the wrong edge statistics.
3. **Re-encode through the same codec** (`reencode_h264=True`). Otherwise the
   model learns "real embers have compression ringing, synthetic ones don't" and
   collapses on real footage.

And: sample from your *measured* distributions, not from uniforms. That is the
most common way synthetic-data pipelines quietly fail.

---

## Stage 5 — the model

### Smoke-test first

```bash
bash scripts/04-submit_job.sh smoke  # on Batch: the first job after building the image
python tests/smoke_train.py          # locally: ~15 s on a GPU, ~60 s on CPU
```

Runs the entire torch path at toy scale: 60 tiny tiles, model construction, a
loss-decreases check, 2 epochs, checkpoint save and reload, **an interrupted
run resumed from `last.pt`**, sliding-window inference on the GPU, and tracks
out the far end. Nine stages, each reporting PASS/FAIL independently so a
failure names the broken piece instead of handing you a traceback.

Run it on every new machine or image **before** starting a long run. A GPU
machine bills from the moment it starts, so finding a broken import 20 minutes
into training costs money and momentum.

It also matters for a specific reason: PyTorch could not be installed in the
environments where this kit was written or rebuilt for Batch, so `model.py`,
`train.py` and `infer.py` (and the resume logic) have never actually executed.
Their tensor shapes and parameter count are verified arithmetically and
everything around them is tested end to end, but the first real execution is
the smoke job.


```python
from firebrand import train as TR
model, hist = TR.train(X, Y, cfg.train, out_dir="runs/streak")
```

Two decisions do most of the work, and neither is the architecture.

**The input is time, not colour.** Channels are residual frames at t−1, t, t+1
rather than R, G, B. A network given one frame is being asked "is this bright
thing static?" from an image that cannot answer it — the question isn't hard, it
is underdetermined. This is a dataloader change and it is the single biggest
lever in the pipeline.

**The output is a mask, not a box.** Regressing four box coordinates to
sub-pixel precision from a 7-pixel object is numerically unstable, and IoU ≥ 0.5
is meaningless at that size — a one-pixel error already fails it. A dense mask
is well-conditioned, needs less data, and gives you streak length and
orientation directly, which you need for the physics anyway.

The network is a 3-level U-Net, ~0.48M parameters at `base_ch=16`. That is
plenty: this is a low-level texture task, not a semantic one. Bigger backbones
overfit a small real set and train slower for no gain.

**Colab:** ~8k tiles of 256 px at batch 16 fits a free-tier T4 and runs 30
epochs in 15–25 minutes. On CPU, drop to `tile=192, epochs=12`.

Watch `val_f1`, not loss — loss on a 1-in-1500 class is dominated by background
and will keep improving while detection quality doesn't. `pos_weight` is set
automatically from the actual imbalance in your tiles (~40 here); leaving it at
1 makes the model predict nothing while showing an excellent loss curve.

### If you have to use YOLO

The lab's README commits to YOLO. It can work, but not out of the box:

- **P2 head** (stride 4, e.g. `yolo11-p2.yaml`). Default `imgsz=640` letterboxes
  1920×1080 by 3×, turning a 7 px streak into 2.3 px — a quarter of one stride-8
  grid cell, below what the architecture can represent.
- **512×512 tiles at native resolution**, SAHI-style tiling at inference. Never
  resize the frame.
- `mosaic=0.0` — mosaic augmentation downsamples and erases objects this small.
- Feed the **3-frame residual stack** in place of RGB. Change the dataloader,
  not the architecture.

---

## Stage 6 — evaluation

```python
from firebrand import evaluate as E
E.print_report(E.report(dets, tracks, gt))
```

**Do not use mAP.** At 7 px, a one-pixel centroid error drops IoU below 0.5, so
`mAP@0.5` reports a genuinely good detector as a failure and you end up tuning
against quantisation noise.

Instead:

- Match on **centre distance ≤ 5 px**.
- Score **tracks**, not frames. A tracker that splits one ember into four scores
  fine on any per-frame metric while ruining your saltation statistics —
  `fragmentation` is the column to watch.
- **Ballistic residual** (`tracks.csv`) is a validation signal that needs no
  labels at all: fit gravity + drag to each track and report RMS residual. ID
  switches between two different embers can't be fit by one parabola and show up
  as large residuals. Most tracking projects don't have a free ground-truth
  proxy like this. Sort by it and inspect the worst.

---

## Stage 7 — pixels to metres

Not implemented here, but the pipeline outputs what it needs.

**The driveway pavers are your calibration target.** They are a regular grid of
known-size rectangles lying in the ground plane — measure a few on site and
`cv2.findHomography` gives you a ground-plane calibration with no checkerboard.
Undistort the fisheye first; the straight paver joints also let you
self-calibrate the distortion from line-straightness alone.

This matters because **saltation contacts happen on the ground plane**, so every
impact and rebound point converts to real metres exactly, with no depth
ambiguity. With one camera the hops in between need an assumption: constrain
each hop to a vertical plane containing the local wind direction (your WindNinja
run gives you that), anchored at the two ground-contact points you can locate
exactly.

Per-hop quantities to log: hop length, apex height, impact and rebound angles,
restitution *e* = v_out/v_in, surface residence time, and integrated brightness
as a size/temperature proxy.

---

## Layout

```
firebrand/
  config.py     every tunable, with the reasoning for each default
  frames.py     decode, frame sources, ROI masks, pre-flight audit
  detect.py     stabilise -> blue channel -> top-hat -> rolling median -> streaks
  track.py      linking, step-ratio self-calibration, acceptance, ballistic check
  review.py     contact sheets and rejection handling
  synth.py      physically-grounded streak rendering and clip generation
  dataset.py    residual stacks, tiling, augmentation (torch-free builder)
  model.py      TinyUNet + weighted BCE/Dice loss
  train.py      training loop
  infer.py      sliding-window inference -> detections -> tracks
  evaluate.py   distance matching, track metrics, detectability filtering
make_mask.py     click-to-draw exclusion mask editor
run_pipeline.py  CLI for stages 1-3, one clip
run_all.py       every stage, resumable, shardable -- what Batch jobs run
batch_entry.py   container entrypoint: bucket paths, logs, hardware record
tests/benchmark.py   end-to-end benchmark and ablation
tests/smoke_train.py torch path check, including resume (the `smoke` step)

Cloud Batch (the lab's gcp-example workflow)
.env.example             project, bucket, region, machines (copy to .env)
Dockerfile               the one image every job runs
requirements-batch.txt   what the image installs
templates/               job.gpu / job.cpu Batch job templates
scripts/00-06            setup, build, upload, submit, sync, watch
scripts/render_job.py    the step menu; fills a template

cloud/                   the old Workbench notebook route
notebooks/Firebrand_Colab.ipynb
```

---

## Known limits

- **Recall after acceptance is ~0.45** on the benchmark, against ~0.75 raw. That
  is by design — requiring three coherent frames costs detections and buys
  precision — and it is the reason to train the model at all: the network sees
  three frames at once and can recover faint, short streaks the shape gate
  rejects.
- **Track stitching is off by default.** `track.stitch()` exists and rejoins
  fragments, but on the benchmark (fragmentation already 1.12) it lost more
  tracks than it merged. Turn it on if your fragmentation exceeds ~1.3.
- **The benchmark is synthetic.** It proves the mechanics work and makes the
  ablations honest. It does not predict field accuracy, because real ember
  brightness and shape distributions differ from the generator's priors. Fit the
  generator to verified real tracks before trusting the numbers.
- **Everything photometric was tuned on one frame** (Left Front, 00:05:36).
  Re-run the audit on your own clips.
