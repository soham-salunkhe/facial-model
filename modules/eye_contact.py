"""
modules/eye_contact.py
━━━━━━━━━━━━━━━━━━━━━━
Eye contact detection, blink counting, and gaze direction estimation.

Responsibilities
----------------
- Compute Eye Aspect Ratio (EAR) per eye using the Soukupova–Čech formula.
- Detect blinks via consecutive-frame EAR threshold crossings.
- Estimate gaze direction (Center / Left / Right / Up / Down) using iris
  centroid position relative to the eye bounding box.
- Track eye contact percentage over a rolling time window.
- Track blink rate (blinks per minute) over a rolling time window.
"""

from __future__ import annotations

import time
from collections import deque
from typing import NamedTuple

import numpy as np

from .landmark_extractor import LandmarkGroups
import config as cfg


class EyeContactResult(NamedTuple):
    """All eye-related metrics for one frame."""
    left_ear:           float   # Eye Aspect Ratio for left eye  (0–1)
    right_ear:          float   # Eye Aspect Ratio for right eye (0–1)
    avg_ear:            float   # Mean EAR
    is_blinking:        bool    # Currently in a blink
    blink_count:        int     # Total blinks since start
    blink_rate:         float   # Blinks per minute (rolling window)
    gaze_direction:     str     # "Center" | "Left" | "Right" | "Up" | "Down"
    eye_contact_pct:    float   # 0–100 percentage of time looking at camera
    gaze_left:          float   # 0/1 boolean-like
    gaze_right:         float
    gaze_up:            float
    gaze_down:          float


def _euclidean(p1: np.ndarray, p2: np.ndarray) -> float:
    return float(np.linalg.norm(p1 - p2))


def _eye_aspect_ratio(pts: np.ndarray) -> float:
    """
    Compute EAR from 6 eye landmark points.

    pts[0] = p1 (left corner)
    pts[1] = p2 (upper-left vertical)
    pts[2] = p3 (upper-right vertical)
    pts[3] = p4 (right corner)
    pts[4] = p5 (lower-right vertical)
    pts[5] = p6 (lower-left vertical)

    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    """
    vertical_1 = _euclidean(pts[1], pts[5])
    vertical_2 = _euclidean(pts[2], pts[4])
    horizontal = _euclidean(pts[0], pts[3])
    if horizontal < 1e-6:
        return 0.0
    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def _gaze_direction(
    iris_center: np.ndarray,
    eye_bbox: tuple,
) -> tuple[str, float, float, float, float]:
    """
    Determine gaze direction from iris centroid position within eye bbox.

    Parameters
    ----------
    iris_center : np.ndarray  shape (2,)  – pixel (x, y) of iris center
    eye_bbox    : (x_min, y_min, x_max, y_max) of eye contour

    Returns
    -------
    direction, gaze_left, gaze_right, gaze_up, gaze_down
    """
    x_min, y_min, x_max, y_max = eye_bbox
    eye_w = max(x_max - x_min, 1.0)
    eye_h = max(y_max - y_min, 1.0)

    # Normalise iris position within eye bounding box (-0.5 … 0.5)
    rel_x = (iris_center[0] - x_min) / eye_w - 0.5
    rel_y = (iris_center[1] - y_min) / eye_h - 0.5

    thresh = cfg.GAZE_CENTER_THRESHOLD

    gaze_left  = 0.0
    gaze_right = 0.0
    gaze_up    = 0.0
    gaze_down  = 0.0

    if abs(rel_x) > abs(rel_y):
        if rel_x < -thresh:
            gaze_left = 1.0
            return "Left", gaze_left, gaze_right, gaze_up, gaze_down
        elif rel_x > thresh:
            gaze_right = 1.0
            return "Right", gaze_left, gaze_right, gaze_up, gaze_down
    else:
        if rel_y < -thresh:
            gaze_up = 1.0
            return "Up", gaze_left, gaze_right, gaze_up, gaze_down
        elif rel_y > thresh:
            gaze_down = 1.0
            return "Down", gaze_left, gaze_right, gaze_up, gaze_down

    return "Center", gaze_left, gaze_right, gaze_up, gaze_down


class EyeContactAnalyzer:
    """
    Stateful analyzer tracking blinks, gaze direction, and eye contact.

    Designed to be called once per frame with the current LandmarkGroups.

    Parameters
    ----------
    fps : float
        Expected frames-per-second (used for time-based windows).
    """

    def __init__(self, fps: float = 30.0) -> None:
        self._fps = fps
        self._consec_blink_frames = 0
        self._total_blinks = 0
        self._is_blinking = False

        # Session start — used to compute elapsed time for blink rate window.
        # time.monotonic() returns system uptime (e.g. 48,000 s), NOT session time.
        # Using it directly as the denominator caused blink rate to be underreported
        # for the first 60 seconds of every new session.
        self._session_start: float = time.monotonic()

        # Rolling timestamps for rate calculations
        _eye_window = cfg.EYE_CONTACT_WINDOW_SEC
        _blink_window = cfg.BLINK_RATE_WINDOW_SEC
        self._eye_contact_events: deque[tuple[float, bool]] = deque()
        self._blink_timestamps:   deque[float]              = deque()

        self._eye_contact_window = _eye_window
        self._blink_rate_window  = _blink_window

    def analyze(self, groups: LandmarkGroups) -> EyeContactResult:
        """
        Run all eye-related analysis on the current frame's landmark groups.

        Parameters
        ----------
        groups : LandmarkGroups
            Output of LandmarkExtractor.extract()

        Returns
        -------
        EyeContactResult
        """
        now = time.monotonic()

        # ── EAR computation ─────────────────────────────────────────────
        left_ear  = _eye_aspect_ratio(groups.left_ear_pts)
        right_ear = _eye_aspect_ratio(groups.right_ear_pts)
        avg_ear   = (left_ear + right_ear) / 2.0

        # ── Blink detection ─────────────────────────────────────────────
        currently_below = avg_ear < cfg.EAR_BLINK_THRESHOLD
        if currently_below:
            self._consec_blink_frames += 1
        else:
            if self._consec_blink_frames >= cfg.EAR_CONSEC_FRAMES:
                # Rising edge — completed blink
                self._total_blinks += 1
                self._blink_timestamps.append(now)
            self._consec_blink_frames = 0

        self._is_blinking = currently_below and (
            self._consec_blink_frames >= cfg.EAR_CONSEC_FRAMES
        )

        # Prune old blink timestamps
        cutoff = now - self._blink_rate_window
        while self._blink_timestamps and self._blink_timestamps[0] < cutoff:
            self._blink_timestamps.popleft()

        # Blinks per minute — use ELAPSED SESSION TIME as denominator, not system uptime.
        # Cap at the configured window (60 s) so rate isn't under-diluted early in session.
        elapsed = now - self._session_start
        window_secs = min(max(elapsed, 1.0), self._blink_rate_window)
        blink_rate = (len(self._blink_timestamps) / window_secs) * 60.0

        # ── Gaze direction ──────────────────────────────────────────────
        # Average left and right iris positions
        left_iris_c  = groups.left_iris_center
        right_iris_c = groups.right_iris_center

        left_dir, gl, gr, gu, gd = _gaze_direction(
            left_iris_c, groups.eye_bbox("left")
        )
        right_dir, _gl, _gr, _gu, _gd = _gaze_direction(
            right_iris_c, groups.eye_bbox("right")
        )

        # Agree on direction: if both eyes agree, use that; else Left/Right wins
        if left_dir == right_dir:
            direction = left_dir
            gaze_l, gaze_r, gaze_u, gaze_d = gl, gr, gu, gd
        elif left_dir == "Center":
            direction = right_dir
            gaze_l, gaze_r, gaze_u, gaze_d = _gl, _gr, _gu, _gd
        elif right_dir == "Center":
            direction = left_dir
            gaze_l, gaze_r, gaze_u, gaze_d = gl, gr, gu, gd
        else:
            direction = left_dir
            gaze_l, gaze_r, gaze_u, gaze_d = gl, gr, gu, gd

        # ── Eye contact tracking ────────────────────────────────────────
        is_contact = (direction == "Center") and (not self._is_blinking)
        self._eye_contact_events.append((now, is_contact))

        ec_cutoff = now - self._eye_contact_window
        while self._eye_contact_events and self._eye_contact_events[0][0] < ec_cutoff:
            self._eye_contact_events.popleft()

        total_events = len(self._eye_contact_events)
        contact_events = sum(1 for _, c in self._eye_contact_events if c)
        eye_contact_pct = (contact_events / max(total_events, 1)) * 100.0

        return EyeContactResult(
            left_ear        = left_ear,
            right_ear       = right_ear,
            avg_ear         = avg_ear,
            is_blinking     = self._is_blinking,
            blink_count     = self._total_blinks,
            blink_rate      = blink_rate,
            gaze_direction  = direction,
            eye_contact_pct = eye_contact_pct,
            gaze_left       = gaze_l,
            gaze_right      = gaze_r,
            gaze_up         = gaze_u,
            gaze_down       = gaze_d,
        )

    def reset(self) -> None:
        """Reset all counters and history (e.g., between interview sessions)."""
        self._consec_blink_frames = 0
        self._total_blinks = 0
        self._is_blinking = False
        self._eye_contact_events.clear()
        self._blink_timestamps.clear()
        # Restart the session clock so blink rate is accurate from the new session start
        self._session_start = time.monotonic()
