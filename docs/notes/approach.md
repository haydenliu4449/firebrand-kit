# Firebrand detection — approach notes

> Moved here on 2026-11-05 from the claude.ai project "Firebrands", where it was
> kept during the first two months of work. This copy is the one to keep
> current: add a numbered finding for each change that alters the plan, with
> what was measured and how it was verified.

Full write-up with figures and code sketches:
https://claude.ai/code/artifact/272369a6-4198-4048-9a56-6901ef3af3db

Working code: `firebrand-kit` (GitHub repo; runs on Cloud Batch — see
`docs/notes/batch-workflow.md`). Run `python tests/benchmark.py --ablate --channels`
to reproduce the benchmark numbers below.

Context: preliminary CV model to tag firebrands in 10–60 s training clips.
Prior attempt (OpenCV auto-label → YOLO) underperformed. **Single camera view
confirmed — no overlapping siblings, so no triangulation.** Clips are 3840×2160
at 24 fps (clip5 checked).

## Core reframing

A firebrand is defined by its **motion**, not its appearance. In a single frame
a 7 px glowing ember and a 7 px paver joint are the same object. Detection must
be temporal (track-before-detect), and the tracker itself becomes the label
source: a candidate is accepted only if it joins a ≥3-frame track with smooth
heading and consistent brightness.

## Benchmark results (implemented and measured)

50 frames of synthetic firebrands composited additively in linear light onto a
real frame from the camera, with 6% flicker and 0.8 px camera shake. Since
2026-10-08 the benchmark also runs the same clip mirrored left-right (every
ember flies the other way); the two rows must match.

| Configuration | Result |
|---|---|
| **Shipped defaults (2026-10-08)** | **P = 1.00**, raw recall 0.74. After tracking: P = 1.00, recall 0.72; track recall 0.69, fragmentation 1.22, 2 ID switches. **Mirrored: identical.** |
| Defaults 2026-09-28 → 10-07 | After tracking P = 1.00, recall 0.70; track recall 0.62. **Mirrored: 0 tracks** (finding 5) |
| Defaults before 2026-09-28 | P = 1.00, raw 0.74 → 0.45 after tracking; track recall 0.54, frag 1.14, 0 ID switches |
| No stabilisation | P = 0.08 raw (2876 det); after tracking P 0.17 (0.19 with the old one-ended linker) |
| No edge suppression | P = 0.17 raw (1413 det) |
| Neither | P = 0.06 (4114 det) — this is the v1 failure mode reproduced |

Precision 1.00 is the number that matters: it makes auto-labels safe to train on.
The synthetic generator only launches embers rightward (heading −40°…85°) and
at 3–10 embers per frame, so the benchmark neither tested direction (until the
mirrored row) nor dense showers (still untested — see finding 5).

## Findings that changed the plan

**1. Stabilisation is essential and was missing from the first draft.**
The camera is house-mounted in a wind event. A *sub-pixel* tremor breaks
rolling-median subtraction — a moving high-contrast edge reappears in the
residual as a bright elongated streak-shaped object, present in every frame.
Fix: `cv2.phaseCorrelate` sub-pixel translation alignment (translation only —
rotation estimation is unstable on smoke-filled frames), plus a per-pixel
threshold surcharge proportional to |∇background| (`edge_suppress`), since
misalignment residual scales with scene gradient and firebrands don't care where
scene edges are. (clip5 measured shift is only ~0.07 px, so detections drawn on
raw frames are not offset.)

**2. The blue-channel advantage is narrower than first stated — and more
important than it looked.** Single-frame separability (red 3.6σ, gray 9.0σ,
blue 19.7σ) is largely redundant once background subtraction works:

| ember brightness | blue recall | gray recall | blue trkR | gray trkR |
|---|---|---|---|---|
| bright (70–255) | 0.73 | 0.72 | 0.42 | 0.42 |
| medium (35–90) | 0.80 | 0.51 | 0.60 | 0.40 |
| faint (20–55) | 0.71 | **0.19** | 0.64 | **0.00** |

Blue buys nothing on bright embers, and is the difference between working and
not working on faint ones. That is the population that matters: **an ember is
dimmest while it is on the ground losing heat, and ground contact is the
saltation event.** Keep blue.

**3. (2026-09-28) On real footage the tracker, not the detector, was losing the
bright firebrands.** Symptom: first model's tags had no false positives but
missed many firebrands. Measured on 40 frames of clip5 (4K): 181 tracks
accepted vs 311 coherent tracks (3–16 frames) rejected — 193 by the brightness
ratio (in the dense ember storm the linker hops from a bright firebrand onto an
adjacent faint streak, and one bad link discarded the whole firebrand), 118 by
heading (106 of them one bend of 27–31°). Rejected tracks held the bright, wide
embers (peak p90 112 vs 51; width p90 15 vs 8 px). Consequence: the generator,
fitted to accepted tracks, capped synthetic brightness at ~80, so the model never
learned bright firebrands. Pixel limits (max length/area/step) were also tuned at
1080p (minor: 0.4% of candidates). Fixes: cut tracks at the bad link and judge
pieces (`track.split_bad_links`), heading 35°, whole-track brightness ratio 10,
frame-to-frame jump 5 as the cut rule, pixel limits scaled by width/1920,
generator brightness upper bound from every coherent track in the footage,
psf width scaled. clip5 classical accepted tracks 181 → 421; added tracks looked
like embers on contact sheets. Also fixed: `work/config.json` from an old tiles
run had been silently overriding new detect/track defaults. Diagnostic added:
tag overlay draws model detections the tracker dropped in yellow
(`candidates.csv`) — yellow on a miss = tracker, nothing = model/threshold.
(User committed these 2026-10-01 and retrained.)

**4. (2026-10-02) After retraining, the model tagged texture whenever a clip
went quiet.** Cause: the model had never seen a scene with nothing in it.
Every synthetic frame had 2–14 embers, and the backgrounds were the first 12
frames of only the first ~4 clips alphabetically (40-frame cap) — mid-storm
frames full of real embers labelled "not a firebrand" — concatenated so the
scene jumped between clips every 12 frames (whole-scene residual junk for
~half a median window after each jump). Fixes: `work/negatives.txt` lists
user-confirmed firebrand-free frame ranges; the tiles stage adds real all-zero
tiles from them (25% of the set, `--neg-frac`) and uses them as clean synthetic
backgrounds (half the chunks); backgrounds now come from every clip, one clip
per chunk, played ping-pong so they never jump; the eval stage counts false
tracks per 100 firebrand-free frames at every threshold and picks the best-F1
threshold under `--max-empty-fp` (1.0); tag has `--min-track-conf`
(hysteresis: keep a track only if one detection reaches that probability) as a
no-retrain stopgap. `suggest_negatives.py` proposes ≥2 s stretches with no
classical track (with video times; flags model tracks inside them) for the user
to confirm by watching. Rule: never list a frame containing even a faint real
ember — that teaches faint embers are background.

**5. (2026-10-08) The linker could not follow an ember flying left — and in
dense showers tracks hop between embers.** With one detection, `Track.predict`
extrapolates along the streak (step = L × step_ratio), but a streak's angle is
defined mod π and the code tried only the +θ end. minAreaRect θ lies in
[−90°, 90°), so +θ always pointed right: a left-flying ember never got its first
link and every detection became a rejected 1-frame track. This also applied to the model's
tags (same linker). Benchmark mirrored: track recall 0.00 → 0.69 after the fix
(right-moving 0.62 → 0.69 too). clip5, 50 frames, same detections: 589 → 657
accepted tracks, left-moving 9 → 419 (64% of the scene's embers fly left),
ballistic residual median 2.16 → 1.44 px, 1-frame leftovers 1090 → 569. Fix:
both ends are candidates; the linker takes the nearer. No retrain needed
(training flips/rotates tiles, so the model sees all directions); re-tag.
Cost: in clutter (benchmark without stabilisation) ~12–15% more noise tracks;
shipped settings keep P = 1.00; the eval empty-frame threshold absorbs it at
the next train.
**Still open:** drawing a track's boxes on the max of its frames showed many of
the old linker's rightward clip5 tracks chaining from ember to ember across the
shower (steps ≈ 5.8 × L vs the calibrated 4.9; worse axis alignment). They passed
every acceptance rule because shower neighbours fly parallel at similar speeds.
In a random sample of the *new* tracks, about half still hop. **Contact sheets
cannot reveal this** — each crop is a real ember. Proposed next fix: gate every
link on streak geometry (displacement along both streaks' axes, |step| ≈
step_ratio × L), plus a path view (max-projection with boxes) in the review
sheets, plus a dense synthetic benchmark to measure it. Also seen: on a busy
frame the 6σ threshold (≈7 on residual where bright streaks peak at 100–200)
makes bright embers' blobs fat (W 14–32 px) and merges neighbours, which feeds
both the hopping and loose boxes on the classical side.

**6. (2026-10-08) Boxes follow the streak.** Overlay and YOLO labels used to be
upright squares of side max(L, 6) + pad, ignoring θ and W (median 3.2× the area
of the streak's own box on real clip5 detections, 4.4× on synthetic). Now
`detect.box_points` gives the minAreaRect box (L + 1 + 2·pad along θ, W + 1 +
2·pad across; +1 because minAreaRect measures between pixel centres). Overlay
draws it 4 px clear so the box never hides the ember; tag writes `labels/`
(tightest upright YOLO box, 2 px pad) and `labels_obb/` (YOLO OBB corners). For
analysis use x, y, L, W, θ from detections.csv directly. Model detections fit
tightest (trained target is a 2 px line); classical boxes on bright embers are
looser (finding 5, fat blobs).

**7. (2026-10-22) A long firebrand-free video needs sampling, a held-out
split and a check — the first negatives code used it badly.** User has a
30-min video with many firebrand-free stretches. The 10-02 code would have
taken the first 600 listed frames (~25 s of a 30-min file), decoded from frame 0
for every range (quadratic on a long file), and measured "false tracks per 100
empty frames" on the first 200 of those same training frames (optimistic →
threshold too low). Rewritten in `firebrand/negatives.py`:
- negatives.txt accepts **times** (`m:ss`, `h:mm:ss`; 0.25 s trimmed off each
  end of a time range) as well as frames; long videos go in **`negclips/`**,
  which detect/tag never read (`--neg-clips`; 03 uploads it; Batch passes
  /mnt/disks/share/negclips).
- Listed footage → 32-frame blocks on the clip's own frame grid; **every 4th
  block held out** for eval (role fixed by position, so tiles and eval always
  agree and editing negatives.txt never reshuffles). Train takes the middle 16
  frames of blocks in **van der Corput order** (any prefix spread over the
  whole footage): defaults 640 train frames (40 moments), 320 eval frames.
- **Seek** with `CAP_PROP_POS_FRAMES` (exact on this camera's HEVC, keyframe
  every ~2 s: clip5 and a 3-min test file, 45/45 random seeks, ~0.04 s each;
  falls back to grab-forward if the reported position disagrees).
- **Classical check per block**: detect + link + accept on the block; any track
  → block dropped and logged with video time. Test video (4 min, synthetic
  embers only in 2:00–2:20): all 11 train blocks inside dropped, the 6 around
  kept, no false drops on the quiet part; dense real clip5 block dropped.
- **Hard negatives**: with a previous best.pt and a GPU, the model runs on every
  4th kept frame and tiles are cut where it fires (≥ `--mine-thresh` 0.3), up
  to half of the real negatives; log reports "fired on N of M frames".
- Clean synthetic backgrounds now come from checked blocks spread over the
  footage, loaded lazily (were the first ranges, held in RAM).
- Train blocks read in worker processes (spawn; falls back to serial if
  workers fail); eval reads serially. Cost ~2–4 s CPU per 4K frame (median
  ~1.0 s + edge map ~0.6 s on 2 vCPU) → defaults add ~20–40 min to a train job
  on g2-standard-4.
- Seeked stacks equal whole-clip stacks except for the stabiliser's reference
  frame (identical with stabilisation off) — same as the old code.
- Verified with a torch stand-in (no torch in the dev container): tiles stage
  end to end with 2 workers, hard-negative path, eval stage. `tests/test_negatives.py`
  covers parsing, split disjointness/stability, spread.

**8. (2026-10-23) The model had never trained on a real firebrand — and was
taught real ones as background.** User asked how to use the tagged clips to tag
more firebrands. Found: verified tracks only tuned the generator; all positives
were synthetic; and half the synthetic scenes were painted on clip frames full
of unlabelled real embers labelled 0 (directly against recall). Added
`--real-positives` (`firebrand/reallabels.py`):
- Labels = verified classical tracks (minus clips reject.txt) + model tag
  tracks with max prob ≥ `--real-min-conf` 0.8, minus `work/tags/<clip>/reject.txt`
  (from evaluate_tags model-only review). Same ember in both (≤ 5·px_scale px) →
  model's thinner fit; W > 6·px_scale → not a label. Target = 2 px line along
  the core, length L − W (measured streak = core + blur), like synth.gt_masks.
- **Unlabelled embers removed from the labelled (centre) frame, per pixel:**
  keep each label's box + 4 px; zero every core ≥ 2 px above median + 4.5σ
  outside those boxes, with its glow (> 3σ within 4 px). Per-blob judging failed
  on clip5: at 3σ (and even 4.5σ) a bright ember's glow merges with neighbouring
  labelled streaks (a 746 px region) and was kept whole. Per-pixel: bright
  pixels outside labels 1398 → 14 in a busy clip5 tile, labelled streaks
  untouched, 95% of faint grain unchanged (removing all specks would teach
  "any speck left is an ember"). clip5 is dense: ~1400 blobs > 3σ per 4K frame
  vs 33–50 labels.
- Held-out: every 4th 32-frame block (`negatives.is_heldout`, shared with the
  negatives split) never trained on; `evaluate_tags.py --heldout` scores only
  those → fair before/after (writes summary_heldout.csv).
- Mix: `--real-frac` 0.3 of the set, `--real-frames` 640 (~20–30 min extra).
- Tag stage now moves an old tags reject.txt aside (IDs renumber); a review
  older than tags/done.json is ignored; 03 uploads tags reject.txt.
- Also (default): with ≥ 4 checked firebrand-free runs, synthetic embers are
  painted only on those (up to 8), not on clip frames.
- Verified torch-free: tiles stage on clip5 + fake tags (264 real tiles, 2475
  labelled embers), held-out scoring on shifted frames, tests/test_reallabels.py.
  Not verified: effect on the real model (needs a Batch train run).
- Risk: confirmation bias — unreviewed model false positives become positives.
  Guard: review model-only sheets; or `--real-min-conf 0.9`. Helps "no box"
  misses only; yellow-box misses are the tracker.

## Pipeline (as implemented in firebrand-kit)

0. Decode with `ffmpeg -vsync 0`; audit for DVR-padded duplicate frames (a 0.5
   duplicate fraction means every velocity is wrong by 2×) and IR night mode.
1. ROI mask: timestamp, "Left Front" label, **car windscreen/hood** (reflections
   create mirrored duplicate firebrands → double-counted flux).
2. **Stabilise** (phase correlation) + normalise global gain.
3. Blue channel → morphological top-hat.
4. Rolling-median background subtraction (~11 frames) + gradient-proportional
   edge suppression.
5. Connected components → `minAreaRect` → (x, y, L, W, θ, peak, flux).
   **speed = L / t_exposure** gives instantaneous velocity from a single frame.
6. Link on predicted position (IoU trackers give zero overlap at these speeds;
   `linear_sum_assignment` per frame pair; a 1-detection track predicts at
   **both** ends of its streak). Cut at bad links, then accept on ≥3 frames +
   smooth heading + consistent brightness + bounded speed CV.

**Step-ratio self-calibration:** exposure time is not a blocker. For each
streak, look at next-frame detections along its own direction (mod π) and
histogram distance ÷ streak length — genuine successors agree, accidental ones
scatter. Recovers a sharp estimate with no camera metadata (clip5: 4.9). Still
want real exposure for *absolute* speeds (the recovered ratio uses measured
lengths, which include PSF broadening, so it's self-consistent for prediction
but biased physically).

## Labelling under a few-hours budget

- **Contact sheets**: one decision per track, not per box (~25× faster). Real
  firebrands sit still in the centre of every crop while the background slides
  past. 24 tracks/sheet, ~30 s/sheet. Review guidance given to user: keep faint
  embers and tracks with neighbouring embers in the crop; reject static texture
  and tracks whose centre hops between different embers. **Limit:** hopping
  between parallel shower embers is invisible on a sheet (finding 5).
- **Synthetic generation**: firebrand = moving point emitter integrated over the
  exposure. Composite *additively, in linear light*, onto *real* backgrounds,
  then *re-encode through the same H.264 settings*. Sample from measured
  distributions (`synth.fit_config_to_tracks`), never uniforms — but measure
  them on the footage, not only on what a conservative tracker kept (finding 3).
- **Real negatives**: confirmed firebrand-free frames (finding 4) are the only
  labels that are certain without drawing anything.
- **Rejected tracks are free hard negatives** — keep `negatives.csv` (not yet
  used in training: tiles around them in a dense storm would also contain
  unlabelled real embers).
- Re-running detect renumbers tracks: old reject.txt is moved aside
  automatically; reviews must be redone on new sheets.

## Model

- Input: 3-frame **residual stack** (t−1, t, t+1), not RGB. Dataloader change,
  biggest single lever.
- Output: per-pixel streak **mask**, not boxes. TinyUNet, ~0.48M params at
  base_ch=16. Box regression on 7 px objects is unstable; the mask also yields
  L and θ directly for the physics.
- `pos_weight` auto-set from actual imbalance (~40; streak pixels are ~1 in
  1500). Left at 1, the model predicts nothing while showing a great loss curve.
  Side effect: early epochs over-mark (pixel P ~0.1–0.3, R ~0.8); pixel P
  understates track precision (masks wider than the 2 px target).
- Augmentation: rot90 × flips (all directions of travel), gain jitter, noise.
- ~9.5k training tiles, batch 16, ~30 s/epoch on the Batch L4.
- If YOLO is required: P2 head (stride 4), 512×512 native-res tiles, SAHI
  tiling, `mosaic=0.0`, and still feed the residual stack. `labels_obb/` is
  the YOLO-OBB export.

## Evaluation

- Match on **centre distance ≤ 5 px**, not IoU — at 7 px, mAP@0.5 punishes a
  correct detector (even more so with thin oriented boxes).
- Score tracks (HOTA/IDF1, fragmentation), not frames.
- **Ballistic residual** as a label-free check: fit gravity+drag per track; ID
  switches can't be fit by one parabola. Sort `tracks.csv` by this column.
- Real footage: `evaluate_tags.py` — model tracks vs verified classical tracks
  (confirmed / known false / model-only), recall vs verified, precision range
  until model-only sheets are reviewed.
- Empty footage: false tracks per 100 firebrand-free frames (eval stage, needs
  negatives.txt).
- Path view (max of a track's frames with its boxes drawn) is the quickest way
  to see a hopping track; not yet in the review sheets.

## Pixels → metres (single view)

- **The driveway pavers are the calibration target** — known-size ground-plane
  grid → `cv2.findHomography`, no checkerboard. Undistort fisheye first; the
  straight paver joints permit line-straightness self-calibration.
- Saltation contacts lie on the ground plane → impact/rebound points convert
  exactly. Hops between need a vertical-plane assumption aligned to the
  WindNinja wind direction, anchored on the two contact points.
- **Report hop lengths (exact) separately from apex heights (assumption-
  dependent).** Conflating them is the obvious reviewer objection to a
  single-view study.
- Per-hop: hop length, apex height, impact/rebound angle, restitution
  e = v_out/v_in, residence time, integrated brightness as size proxy.

## Known limits

- Recall after tracking 0.72 vs 0.74 raw on the benchmark (0.70 before
  2026-10-08, 0.45 before 2026-09-28); the network recovers the rest.
- Track identity in dense showers is unreliable (finding 5): trajectories from
  dense stretches should not be used for saltation statistics until links are
  gated on streak geometry. Sparse scenes are expected to suffer less (a hop
  needs a neighbour in the gate) — not measured.
- A split at a sharp turn keeps the joint detection with the earlier piece, so a
  bounce yields descent (ending at the contact) + ascent as two tracks — joining
  them per hop is still to do for saltation statistics.
- Embers that sit still for more than ~5 frames fade into the 11-frame rolling
  median; if missed embers are stationary on the ground, try
  `--set detect.median_window=31`.
- `track.stitch()` exists but is **off by default**. Enable if fragmentation
  exceeds ~1.3 (now 1.22 on the benchmark).
- The negatives changes (finding 4) are verified for plumbing only (tiles,
  eval count, tag filter on clip5 with a stand-in model); their effect on the
  real model is measured by the next train → eval report.
- The benchmark is synthetic and sparse: it proves the mechanics and makes the
  ablations honest, but does not predict field accuracy or dense-shower
  identity.

## Open items

- Camera exposure time (for absolute speeds).
- True capture rate vs container fps (container says 24).
- Whether any clip switches to IR night mode mid-way.
- Done by user 2026-10-22/23: negatives video D01_20250107230423 in negclips/
  (3 stretches, ~26 s); first run failed on the old image (times unreadable);
  told to pull + rebuild; a retrain then ran (results not reported yet).
- Next (user, 2026-10-23): pull + rebuild; baseline `evaluate_tags.py --heldout`;
  review model-only sheets → tags reject.txt → 03; `train -- --force
  --real-positives`; re-tag; compare `--heldout` recall/model counts.
- Next (offered): streak-geometry link gating + path view in review sheets +
  dense synthetic benchmark (finding 5).
