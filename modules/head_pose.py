"""
modules/head_pose.py
━━━━━━━━━━━━━━━━━━━━
Head pose estimation using PnP (Perspective-n-Point).

Responsibilities
----------------
- Map 6 facial landmark 2D pixel positions to known 3D model points.
- Solve PnP using OpenCV to get rotation and translation vectors.
- Convert rotation vector → rotation matrix → Euler angles.
- Classify the head orientation into a human-readable label.
"""

from __future__ import annotations

from typing import NamedTuple

import cv2
import numpy as np

from .landmark_extractor import LandmarkGroups
import config as cfg


# ── Generic 3-D face model ──────────────────────────────────────────────────
# Reference 3D coordinates (mm) for 6 key facial landmarks.
# These are canonical values from the Zisserman face model used widely
# in head-pose estimation research.
_MODEL_POINTS_3D = np.array([
    (  0.0,    0.0,   0.0),   # Nose tip
    (  0.0, -330.0, -65.0),   # Chin
    (-225.0,  170.0,-135.0),  # Left eye outer corner
    ( 225.0,  170.0,-135.0),  # Right eye outer corner
    (-150.0, -150.0,-125.0),  # Mouth left corner
    ( 150.0, -150.0,-125.0),  # Mouth right corner
], dtype=np.float64)

# No lens distortion assumed
_DIST_COEFFS = np.zeros((4, 1), dtype=np.float64)


class HeadPoseResult(NamedTuple):
    """Head pose estimation output for one frame."""
    # Smoothed Euler angles (EMA-filtered) — use these for display and classification
    pitch:        float   # Vertical rotation  (+ = looking down, − = looking up)
    yaw:          float   # Horizontal rotation (+ = right, − = left)
    roll:         float   # Tilt rotation       (+ = right tilt, − = left tilt)
    label:        str     # "Stable" | "Looking Left" | "Looking Right" |
                          #  "Looking Up" | "Looking Down" | "Tilted"
    is_stable:    bool    # True when pose is within comfortable interview range
    rvec:         np.ndarray   # Rotation vector (for drawing axis)
    tvec:         np.ndarray   # Translation vector (for drawing axis)
    camera_matrix: np.ndarray  # Camera matrix used
    # Raw (unsmoothed) angles from this frame's solvePnP — for debugging only
    raw_pitch:    float = 0.0
    raw_yaw:      float = 0.0
    raw_roll:     float = 0.0


class HeadPoseEstimator:
    """
    Estimates head pose angles (pitch, yaw, roll) using cv2.solvePnP.

    Maintains a per-session camera matrix computed from the first frame size.
    Call `update_camera(h, w)` if the frame resolution changes.

    Parameters
    ----------
    focal_scale : float
        Focal length expressed as a fraction of image width (default 1.0).
    """

    def __init__(self, focal_scale: float = 1.0) -> None:
        self._focal_scale = focal_scale
        self._camera_matrix: np.ndarray | None = None
        self._last_hw: tuple[int, int] | None = None
        # EMA-smoothed angles — initialised on the first valid estimate.
        # Smoothing reduces Euler angle noise and prevents jerky pose labels
        # caused by per-frame solvePnP instability (especially at moderate yaw).
        self._smoothed_pitch: float | None = None
        self._smoothed_yaw:   float | None = None
        self._smoothed_roll:  float | None = None

    def _build_camera_matrix(self, h: int, w: int) -> np.ndarray:
        """Build an approximate camera intrinsic matrix for unknown cameras."""
        focal = self._focal_scale * w
        cx, cy = w / 2.0, h / 2.0
        return np.array([
            [focal,   0.0, cx],
            [  0.0, focal, cy],
            [  0.0,   0.0,  1],
        ], dtype=np.float64)

    def update_camera(self, h: int, w: int) -> None:
        """Rebuild camera matrix for the given resolution."""
        if self._last_hw != (h, w):
            self._camera_matrix = self._build_camera_matrix(h, w)
            self._last_hw = (h, w)

    def estimate(self, groups: LandmarkGroups) -> HeadPoseResult | None:
        """
        Estimate head pose from landmark groups.

        Parameters
        ----------
        groups : LandmarkGroups

        Returns
        -------
        HeadPoseResult | None
            None if solvePnP fails.
        """
        self.update_camera(groups.frame_h, groups.frame_w)
        camera_matrix = self._camera_matrix

        # 2D image points in pixel space  shape=(6, 2)
        image_points_2d = groups.head_pose_2d.astype(np.float64)

        success, rvec, tvec = cv2.solvePnP(
            _MODEL_POINTS_3D,
            image_points_2d,
            camera_matrix,
            _DIST_COEFFS,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success:
            return None

        # Convert rotation vector → matrix → Euler angles
        rmat, _ = cv2.Rodrigues(rvec)
        raw_pitch, raw_yaw, raw_roll = self._rotation_matrix_to_euler(rmat)

        # Apply EMA temporal smoothing to reduce per-frame noise.
        # On first call, initialise smoothed values to the raw estimate.
        alpha = cfg.HEAD_POSE_SMOOTH_ALPHA
        if self._smoothed_pitch is None:
            self._smoothed_pitch = raw_pitch
            self._smoothed_yaw   = raw_yaw
            self._smoothed_roll  = raw_roll
        else:
            self._smoothed_pitch = alpha * raw_pitch + (1.0 - alpha) * self._smoothed_pitch
            self._smoothed_yaw   = alpha * raw_yaw   + (1.0 - alpha) * self._smoothed_yaw
            self._smoothed_roll  = alpha * raw_roll  + (1.0 - alpha) * self._smoothed_roll

        # Classify using smoothed angles so labels don't flicker
        label, is_stable = self._classify(
            self._smoothed_pitch, self._smoothed_yaw, self._smoothed_roll
        )

        return HeadPoseResult(
            pitch=self._smoothed_pitch,
            yaw=self._smoothed_yaw,
            roll=self._smoothed_roll,
            label=label,
            is_stable=is_stable,
            rvec=rvec,
            tvec=tvec,
            camera_matrix=camera_matrix,
            raw_pitch=raw_pitch,
            raw_yaw=raw_yaw,
            raw_roll=raw_roll,
        )

    @staticmethod
    def _rotation_matrix_to_euler(R: np.ndarray) -> tuple[float, float, float]:
        """
        Decompose a 3x3 rotation matrix into Euler angles (degrees).

        Convention: pitch (X), yaw (Y), roll (Z) in the camera coordinate frame.
        """
        sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        singular = sy < 1e-6

        if not singular:
            pitch = np.degrees(np.arctan2( R[2, 1], R[2, 2]))
            yaw   = np.degrees(np.arctan2(-R[2, 0], sy))
            roll  = np.degrees(np.arctan2( R[1, 0], R[0, 0]))
        else:
            pitch = np.degrees(np.arctan2(-R[1, 2], R[1, 1]))
            yaw   = np.degrees(np.arctan2(-R[2, 0], sy))
            roll  = 0.0

        return float(pitch), float(yaw), float(roll)

    @staticmethod
    def _classify(pitch: float, yaw: float, roll: float) -> tuple[str, bool]:
        """Map Euler angles to a human-readable pose label."""
        if abs(roll) > cfg.ROLL_TILT_THRESHOLD:
            return "Tilted", False
        if yaw < cfg.YAW_LEFT_THRESHOLD:
            return "Looking Left", False
        if yaw > cfg.YAW_RIGHT_THRESHOLD:
            return "Looking Right", False
        if pitch < cfg.PITCH_UP_THRESHOLD:
            return "Looking Up", False
        if pitch > cfg.PITCH_DOWN_THRESHOLD:
            return "Looking Down", False
        return "Stable", True
