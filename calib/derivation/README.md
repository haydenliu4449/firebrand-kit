# How calib/left_front.json was made (2026-11-08)

A record, not a pipeline: these scripts ran once, in this order, from the repository
root, writing into work/calib/ (git-ignored). Decisions taken by hand are listed.

1. Scene: median of 10 frames of clip9 (the clearest clip) -> work/calib/scene_clip9_median.png.
2. Joint map: black-hat of the blue-ish grey image, 17 px ellipse -> ridge.npy (in an inline step).
3. `distortion.py` -- first try: score straightness by a line detector over a k sweep.
   FAILED (flat, noisy score in smoky footage); kept for the record.
4. `trace.py X0` (X0 = 1300, 1700, 2000, 2400, 2800, 3200): follow row joints from a scan line.
5. `lens2.py` -- plumb-line fit of the division model on those traces, robust per line:
   k = -0.446, joints straight to 0.81 px (6.78 px uncorrected). Drain-edge traces
   were rejected (they hop between the grate's parallel edges).
6. `undist_lib.py` (k fixed) + inline remap of the joint map -> ridge_undist.npy.
7. `grid.py` (Hough), `grid2.py` (refit to joint pixels, support filter, merge, order).
   Kept row lines 0-6: gaps 398, 344, 290, 242, 201, 169 px (ratio ~1.19 each, i.e.
   even 0.6 m spacing in perspective). DROPPED 7 and 8 by hand (gaps 101 then 188).
   Hough cross lines were unusable (tyre tracks, the label).
8. `cross.py`: cross joints found between neighbouring row lines; the two at
   -21/-25 deg lay on a tyre track and were dropped (phi < -30 kept).
9. `fitH.py`: homography with row joints at y = 0.6 j and both ends of a cross joint
   at the same x. Row joints within 1-4 mm; cross joints on a half-offset lattice of
   slab length 0.3325 fit units (coherence 0.999, 0.6 cm rms, max 1.3 cm).
10. Scale: slab 1.2 x 0.6 m ASSUMED (not measured). The image fixes the right
    angles and the row spacing but not the slab's length/width ratio.
11. Ground x axis flipped so the frame is a map, not a mirror (checked: the
    "Left Front" label reads normally on a top-down render; tests/test_calibration.py).

Checks: street width kerb to kerb at canvas column 4000 = 5.2 m (6.0 m straight down
the column) under the assumed slab size: narrow for a residential street, so either
the slabs are bigger or the street is not in the driveway's plane. Measure one and
put it in calib/left_front.json.
