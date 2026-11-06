"""Embers lying still on the ground.

Everything else in this kit defines a firebrand by its motion, so an ember that
has landed vanishes: the rolling median takes anything that sits still for half
its window into the background, the residual there is zero, and neither the
classical detector nor the model (whose input IS the residual) can see it. On
clip5 that is dozens of bright dots on the road in every frame. Landing and
resting are the end of every saltation path, so they are worth having.

No motion to go on, so this uses the other thing that separates them from the
scene: the camera never moves and the clips come from different moments, so
the scene is what is there in (almost) every clip. Per pixel, the second
smallest of the clips' medians is the scene with nothing on it; a resting ember
is bright, compact and above that.

Pure OpenCV, a pass over the decoded clip: runs anywhere, no model involved.
Output is its own table (resting.csv), not tracks: x, y and how long it stayed.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class RestConfig:
    tophat_ksize: int = 31      # px at 3840 wide, as the detector's
    gap: int = 4                # frames between the samples of one median
    n_samples: int = 5          # samples per median: must stay still 2*gap+1 frames or so
    every: int = 12             # frames between two looks (0.5 s at 24 fps)
    min_peak: float = 60.0      # above the scene, blue channel
    min_blue: float = 110.0     # and this bright in itself (see find)
    max_scene: float = 20.0     # the scene right there must be plain (see find)
    scene_px: int = 8           # px at 3840 wide: how far "right there" reaches
    min_area: int = 4
    max_area: int = 200         # px at 3840 wide: bigger is scene or smoke
    max_extent: int = 26        # px at 3840 wide: longest side of the box
    same_px: float = 4.0        # px at 3840 wide: still the same ember
    min_looks: int = 2          # looks it must be seen in


def tophat(blue, ksize):
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    return cv2.morphologyEx(blue, cv2.MORPH_TOPHAT, k)


def clip_median(path, ksize, n=7):
    """Top-hat of the blue channel at n moments spread over a clip, median."""
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    got = []
    # The container's count runs a few frames past what decodes on these files.
    for fi in np.linspace(5, max(total - 12, 6), n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, f = cap.read()
        if ok:
            got.append(tophat(f[..., 0], ksize))
    cap.release()
    return np.median(np.stack(got), 0).astype(np.uint8) if got else None


def scene_reference(paths, ksize):
    """The scene with nothing on it, from every clip of the camera."""
    meds = [m for m in (clip_median(p, ksize) for p in paths) if m is not None]
    if not meds:
        return None
    st = np.stack(meds)
    # Second smallest, not smallest: one clip darker than the rest (clip11 is)
    # should not set the scene. Fewer than 3 clips: nothing to compare, the
    # smallest is all there is.
    return np.partition(st, 1, axis=0)[1] if len(meds) >= 3 else st.min(0)


def find(blue, scene, cfg: RestConfig, frame, roi=None, scale=1.0, ksize=31):
    """One still blue image -> [(frame, x, y, peak, area)] of resting embers.

    Standing out from the scene is not enough: the clearer the air, the more
    texture shows, and on quiet clip9 gravel and lit concrete stood 60+ above a
    scene taken mostly from smoky clips. A glowing ember is also bright in
    itself, and texture under orange light has little blue to give: blue at the
    spot >= 120 kept 86 of 164 candidates on clip5 and 13 of 454 on clip9.
    (Colour ratios do not help: delta-B/delta-R is ~1 for both.)

    Nor can a dot be told from the scene where the scene itself is strong: wall
    edges, cracks, puddle rims and sunlit leaves change with the light and leave
    a remainder. On 60 tiles judged by eye, scene top-hat < 20 within 8 px kept
    38 of 40 real embers and 8 of 15 false ones, and 93% of all candidates.
    """
    d = cv2.subtract(tophat(blue, ksize), scene)
    if roi is not None:
        d = cv2.bitwise_and(d, d, mask=roi)
    bw = (d > cfg.min_peak * 0.4).astype(np.uint8)
    n, lab, st, ce = cv2.connectedComponentsWithStats(bw, 8)
    out = []
    amax, emax = cfg.max_area * scale * scale, cfg.max_extent * scale
    r = max(1, int(round(cfg.scene_px * scale)))
    for i in np.where((st[:, cv2.CC_STAT_AREA] >= cfg.min_area) &
                      (st[:, cv2.CC_STAT_AREA] <= amax) &
                      (np.maximum(st[:, 2], st[:, 3]) <= emax))[0]:
        if i == 0:
            continue
        x, y, w, h = st[i, :4]
        here = lab[y:y + h, x:x + w] == i
        pk = float(d[y:y + h, x:x + w][here].max())
        if pk < cfg.min_peak or float(blue[y:y + h, x:x + w][here].max()) < cfg.min_blue:
            continue
        cx, cy = int(ce[i][0]), int(ce[i][1])
        if scene[max(cy - r, 0):cy + r + 1, max(cx - r, 0):cx + r + 1].max() >= cfg.max_scene:
            continue
        out.append((frame, float(ce[i][0]), float(ce[i][1]), pk, int(st[i, 4])))
    return out


def group(found, cfg: RestConfig, scale=1.0):
    """Sightings at one spot -> one ember each: rows for resting.csv."""
    live, done = [], []
    tol = cfg.same_px * scale
    gap = 2 * cfg.every
    for f, x, y, pk, a in sorted(found):
        for e in live:
            if f - e["last"] <= gap and f != e["last"] and \
                    np.hypot(e["x"] - x, e["y"] - y) <= tol:
                e["last"] = f
                e["peaks"].append(pk)
                e["areas"].append(a)
                break
        else:
            live.append(dict(x=x, y=y, first=f, last=f, peaks=[pk], areas=[a]))
    done = [e for e in live if len(e["peaks"]) >= cfg.min_looks]
    return [dict(ember_id=i, x=e["x"], y=e["y"], frame_start=e["first"], frame_end=e["last"],
                 looks=len(e["peaks"]), peak=float(np.mean(e["peaks"])),
                 area=float(np.mean(e["areas"]))) for i, e in enumerate(done)]


def run_clip(frames, scene, roi=None, cfg=None, scale=1.0):
    """Stream a clip's BGR frames -> rows for resting.csv. A look at frame f is
    the median of n_samples frames ending at f, labelled by their middle one."""
    cfg = cfg or RestConfig()
    ksize = max(5, int(round(cfg.tophat_ksize * scale / 2)) * 2 + 1)
    buf = deque(maxlen=cfg.n_samples)
    found = []
    for i, f in enumerate(frames):
        if i % cfg.gap:
            continue
        buf.append(f[..., 0].copy())
        if len(buf) == cfg.n_samples and i % cfg.every == 0:
            med = np.partition(np.stack(buf), cfg.n_samples // 2, axis=0)[cfg.n_samples // 2]
            found += find(med, scene, cfg, i - cfg.gap * (cfg.n_samples // 2), roi, scale, ksize)
    return group(found, cfg, scale)
