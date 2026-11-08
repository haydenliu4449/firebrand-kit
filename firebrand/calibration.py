"""Pixels -> metres on the ground, for one camera (calib/<camera>.json).

Two steps, both fitted on the driveway pavers (approach.md finding 12):
  1. lens: the wide-angle lens bends straight lines; a one-parameter division
     model undoes it (fitted so the paver joints come out straight: 0.81 px rms);
  2. ground plane: a homography from the undistorted image to metres on the
     ground, fitted so row joints fall 0.6 m apart and cross joints on a
     half-offset 1.2 m lattice (row joints within 1-4 mm, cross joints 0.6 cm rms).
Only points ON the ground plane convert exactly: ground contacts and resting
embers. An ember in flight is not on the plane and has no ground position from
one camera. The slab size is an ASSUMPTION until measured (slab_size_provisional
in the file); every distance scales with it. The fit is supported over the
driveway; the street is beyond it and assumed to be the same plane.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class GroundCalibration:
    def __init__(self, spec):
        self.spec = spec
        L = spec["lens"]
        self.k, self.c, self.R = float(L["k"]), np.array(L["center"], float), float(L["r_norm"])
        self.H = np.array(spec["H_undistorted_px_to_ground_m"], float)
        self.Hinv = np.linalg.inv(self.H)
        self.provisional = bool(spec.get("slab_size_provisional", True))
        self.region = np.array(spec.get("supported_region_undistorted_px", []), float)

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text()))

    def undistort(self, xy):
        p = np.atleast_2d(np.asarray(xy, float))
        d = (p - self.c) / self.R
        r2 = (d ** 2).sum(1, keepdims=True)
        return self.c + d / (1 + self.k * r2) * self.R

    def distort(self, uv):
        """Inverse of undistort (closed form for the division model)."""
        u = (np.atleast_2d(np.asarray(uv, float)) - self.c) / self.R
        ru = np.hypot(u[:, 0], u[:, 1])[:, None]
        with np.errstate(invalid="ignore", divide="ignore"):
            rd = np.where(ru > 1e-12, (1 - np.sqrt(np.maximum(1 - 4 * self.k * ru ** 2, 0)))
                          / (2 * self.k * ru), 0.0)
            s = np.where(ru > 1e-12, rd / ru, 1.0)
        return self.c + u * s * self.R

    def to_ground(self, xy):
        """Raw pixel(s) -> (N, 2) metres on the ground plane."""
        u = self.undistort(xy)
        q = np.c_[u, np.ones(len(u))] @ self.H.T
        return q[:, :2] / q[:, 2:3]

    def to_pixel(self, XY):
        """Ground metres -> raw pixel(s)."""
        q = np.c_[np.atleast_2d(np.asarray(XY, float)), np.ones(len(np.atleast_2d(XY)))] @ self.Hinv.T
        return self.distort(q[:, :2] / q[:, 2:3])

    def supported(self, xy):
        """True where the pixel lies over the driveway the plane was fitted on."""
        import cv2
        if not len(self.region):
            return np.zeros(len(np.atleast_2d(xy)), bool)
        u = self.undistort(xy).astype(np.float32)
        poly = self.region.astype(np.float32).reshape(-1, 1, 2)
        return np.array([cv2.pointPolygonTest(poly, (float(a), float(b)), False) >= 0 for a, b in u])
