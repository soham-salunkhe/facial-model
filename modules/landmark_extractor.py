"""
modules/landmark_extractor.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Structured semantic landmark extraction from a FaceResult.

Responsibilities
----------------
- Group the raw 478 MediaPipe landmark indices into named semantic clusters.
- Compute derived geometry (bounding boxes, centroids, distances).
- Provide a clean interface consumed by the eye-contact, head-pose,
  smile, and feature-aggregator modules.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional

import numpy as np

from .face_detector import FaceResult
import config as cfg


@dataclasses.dataclass
class LandmarkGroups:
    """Structured semantic landmark data extracted from one frame."""

    # Eye pixel arrays  shape=(N, 2)
    left_eye:  np.ndarray
    right_eye: np.ndarray

    # Iris pixel arrays  shape=(5, 2)
    left_iris:  np.ndarray
    right_iris: np.ndarray

    # Eyebrow pixel arrays  shape=(5, 2)
    left_eyebrow:  np.ndarray
    right_eyebrow: np.ndarray

    # Mouth contour  shape=(12, 2)
    mouth: np.ndarray

    # Nose tip  shape=(2,)
    nose_tip: np.ndarray

    # Jawline  shape=(8, 2)
    jawline: np.ndarray

    # Key EAR-specific points  shape=(6, 2) each
    left_ear_pts:  np.ndarray
    right_ear_pts: np.ndarray

    # Head pose key-point pixel coords  shape=(6, 2)
    head_pose_2d: np.ndarray

    # Full pixel array for motion analysis  shape=(N, 2)
    all_pixels: np.ndarray

    # Frame dims
    frame_h: int
    frame_w: int

    # ── Derived properties ──────────────────────────────────

    @property
    def left_iris_center(self) -> np.ndarray:
        return self.left_iris.mean(axis=0)

    @property
    def right_iris_center(self) -> np.ndarray:
        return self.right_iris.mean(axis=0)

    @property
    def left_eye_center(self) -> np.ndarray:
        return self.left_eye.mean(axis=0)

    @property
    def right_eye_center(self) -> np.ndarray:
        return self.right_eye.mean(axis=0)

    @property
    def mouth_center(self) -> np.ndarray:
        return self.mouth.mean(axis=0)

    @property
    def face_center(self) -> np.ndarray:
        return self.all_pixels.mean(axis=0)

    def eye_bbox(self, side: str) -> tuple:
        """Return (x_min, y_min, x_max, y_max) for left or right eye."""
        pts = self.left_eye if side == "left" else self.right_eye
        return (pts[:, 0].min(), pts[:, 1].min(),
                pts[:, 0].max(), pts[:, 1].max())


class LandmarkExtractor:
    """
    Extracts structured semantic groups from a FaceResult.

    Usage
    -----
    extractor = LandmarkExtractor()
    groups = extractor.extract(face_result)
    """

    # Pre-build numpy index arrays for fast indexing
    _IDX_LEFT_EYE      = np.array(cfg.LEFT_EYE,       dtype=np.int32)
    _IDX_RIGHT_EYE     = np.array(cfg.RIGHT_EYE,      dtype=np.int32)
    _IDX_LEFT_IRIS     = np.array(cfg.LEFT_IRIS,       dtype=np.int32)
    _IDX_RIGHT_IRIS    = np.array(cfg.RIGHT_IRIS,      dtype=np.int32)
    _IDX_LEFT_BROW     = np.array(cfg.LEFT_EYEBROW,   dtype=np.int32)
    _IDX_RIGHT_BROW    = np.array(cfg.RIGHT_EYEBROW,  dtype=np.int32)
    _IDX_MOUTH         = np.array(cfg.MOUTH_OUTER,    dtype=np.int32)
    _IDX_JAWLINE       = np.array(cfg.JAWLINE,         dtype=np.int32)
    _IDX_LEFT_EAR      = np.array(cfg.LEFT_EAR_IDX,  dtype=np.int32)
    _IDX_RIGHT_EAR     = np.array(cfg.RIGHT_EAR_IDX, dtype=np.int32)

    # Head pose solvePnP landmark indices (ordered to match 3D model)
    _HEAD_POSE_IDX = np.array([
        cfg.HEAD_POSE_POINTS["nose_tip"],
        cfg.HEAD_POSE_POINTS["chin"],
        cfg.HEAD_POSE_POINTS["left_eye_outer"],
        cfg.HEAD_POSE_POINTS["right_eye_outer"],
        cfg.HEAD_POSE_POINTS["mouth_left"],
        cfg.HEAD_POSE_POINTS["mouth_right"],
    ], dtype=np.int32)

    def extract(self, face: FaceResult) -> LandmarkGroups:
        """
        Extract semantic landmark groups from a FaceResult.

        Parameters
        ----------
        face : FaceResult
            Output of FaceDetector.detect()

        Returns
        -------
        LandmarkGroups
        """
        px = face.pixels  # shape (N, 2)

        return LandmarkGroups(
            left_eye      = px[self._IDX_LEFT_EYE],
            right_eye     = px[self._IDX_RIGHT_EYE],
            left_iris     = px[self._IDX_LEFT_IRIS],
            right_iris    = px[self._IDX_RIGHT_IRIS],
            left_eyebrow  = px[self._IDX_LEFT_BROW],
            right_eyebrow = px[self._IDX_RIGHT_BROW],
            mouth         = px[self._IDX_MOUTH],
            nose_tip      = px[cfg.NOSE_TIP],
            jawline       = px[self._IDX_JAWLINE],
            left_ear_pts  = px[self._IDX_LEFT_EAR],
            right_ear_pts = px[self._IDX_RIGHT_EAR],
            head_pose_2d  = px[self._HEAD_POSE_IDX],
            all_pixels    = px,
            frame_h       = face.frame_h,
            frame_w       = face.frame_w,
        )
