"""
modules/behavioral_window.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Rolling-window aggregation of per-frame facial analysis signals.

Produces a BehavioralWindowSnapshot with mean/std statistics that summarise
observable behavioral signals over a configurable time window (default 7 s).

Purpose
-------
Frame-level metrics fluctuate too quickly to be meaningful for interview-level
analysis. Aggregating over 5–10 seconds gives a more stable picture of:
  - how consistently the candidate maintained eye contact
  - how stable their head position was
  - average facial expression signals

These are OBSERVABLE / GEOMETRICALLY DERIVED BEHAVIORAL SIGNALS.
They are NOT psychological measurements. Do not interpret them as:
  - proof of confidence, nervousness, deception, or any internal state.
  - clinically validated assessments of any kind.
"""

from __future__ import annotations

import time
from collections import deque
from typing import List, NamedTuple, Optional

import numpy as np

import config as cfg


# ── Per-frame sample ─────────────────────────────────────────────────────────

class FrameSample(NamedTuple):
    """Snapshot of one frame's behavioral metrics for window accumulation."""
    timestamp:        float
    face_detected:    bool
    eye_contact_pct:  float   # 0–100
    head_yaw:         float   # degrees (smoothed)
    head_pitch:       float   # degrees (smoothed)
    head_roll:        float   # degrees (smoothed)
    smile_score:      float   # 0–100
    facial_movement:  float   # 0–3 normalised EMA
    is_head_stable:   bool
    landmark_jitter:  float   # 0–1


# ── Window snapshot ──────────────────────────────────────────────────────────

class BehavioralWindowSnapshot(NamedTuple):
    """
    Aggregated behavioral signals over a rolling time window.

    All values are computed from geometric observations or derived signals.
    Face-absent frames are counted toward face_presence_ratio but excluded
    from facial metric statistics (eye_contact_mean, smile_mean, etc.).
    """
    # Window metadata
    window_duration:      float   # actual elapsed seconds of data in window

    # Eye contact (from face-detected frames only)
    eye_contact_mean:     float   # mean eye contact %  (0–100)
    eye_contact_std:      float   # std dev

    # Head orientation (from face-detected frames only)
    head_yaw_mean:        float   # mean yaw in degrees
    head_yaw_std:         float
    head_pitch_mean:      float
    head_pitch_std:       float

    # Expression / engagement signals
    smile_mean:           float   # mean smile score  (0–100)

    # Motion signals
    facial_movement_mean: float   # mean facial movement (0–3)

    # Stability
    posture_stability:    float   # fraction of face-detected frames with stable pose (0–1)

    # Jitter
    landmark_jitter_mean: float   # mean landmark jitter (0–1)

    # Presence
    face_presence_ratio:  float   # fraction of all frames where a face was detected (0–1)


# ── Aggregator ───────────────────────────────────────────────────────────────

class BehavioralWindowAggregator:
    """
    Accumulates per-frame signals into a rolling time window and provides
    summary statistics via get_snapshot().

    Parameters
    ----------
    window_sec : float
        Duration of the rolling window in seconds. Default from config.
    """

    def __init__(self, window_sec: float = cfg.BEHAVIORAL_WINDOW_SEC) -> None:
        self._window_sec = window_sec
        self._samples: deque[FrameSample] = deque()

    # ── Public API ────────────────────────────────────────────────────────────

    def add_frame(
        self,
        face_detected: bool,
        eye_contact_pct: float = 0.0,
        head_yaw: float = 0.0,
        head_pitch: float = 0.0,
        head_roll: float = 0.0,
        smile_score: float = 0.0,
        facial_movement: float = 0.0,
        is_head_stable: bool = False,
        landmark_jitter: float = 0.0,
    ) -> None:
        """
        Record one frame of metrics into the rolling window.

        When face_detected=False all facial metric arguments are ignored in
        the per-face statistics but the frame still counts toward
        face_presence_ratio and window_duration.
        """
        now = time.monotonic()
        self._samples.append(FrameSample(
            timestamp=now,
            face_detected=face_detected,
            eye_contact_pct=float(eye_contact_pct),
            head_yaw=float(head_yaw),
            head_pitch=float(head_pitch),
            head_roll=float(head_roll),
            smile_score=float(smile_score),
            facial_movement=float(facial_movement),
            is_head_stable=bool(is_head_stable),
            landmark_jitter=float(landmark_jitter),
        ))
        self._prune(now)

    def get_snapshot(self) -> Optional[BehavioralWindowSnapshot]:
        """
        Compute and return the current window snapshot.

        Returns None if no samples have been added yet.
        """
        if not self._samples:
            return None

        now = time.monotonic()
        self._prune(now)

        samples = list(self._samples)
        n = len(samples)
        if n == 0:
            return None

        # Window duration (time from oldest to newest sample in window)
        window_duration = max(
            samples[-1].timestamp - samples[0].timestamp, 0.0
        )

        # Face presence ratio (all frames)
        n_face = sum(1 for s in samples if s.face_detected)
        face_presence_ratio = n_face / n

        # Facial metrics only from face-detected frames
        face_samples = [s for s in samples if s.face_detected]
        nf = len(face_samples)

        if nf == 0:
            # No face data — return zeros
            return BehavioralWindowSnapshot(
                window_duration=round(window_duration, 2),
                eye_contact_mean=0.0,
                eye_contact_std=0.0,
                head_yaw_mean=0.0,
                head_yaw_std=0.0,
                head_pitch_mean=0.0,
                head_pitch_std=0.0,
                smile_mean=0.0,
                facial_movement_mean=0.0,
                posture_stability=0.0,
                landmark_jitter_mean=0.0,
                face_presence_ratio=round(face_presence_ratio, 3),
            )

        eye_vals    = np.array([s.eye_contact_pct for s in face_samples])
        yaw_vals    = np.array([s.head_yaw        for s in face_samples])
        pitch_vals  = np.array([s.head_pitch      for s in face_samples])
        smile_vals  = np.array([s.smile_score     for s in face_samples])
        move_vals   = np.array([s.facial_movement for s in face_samples])
        jitter_vals = np.array([s.landmark_jitter for s in face_samples])
        stable_vals = np.array([float(s.is_head_stable) for s in face_samples])

        return BehavioralWindowSnapshot(
            window_duration=round(window_duration, 2),
            eye_contact_mean=round(float(np.mean(eye_vals)),  2),
            eye_contact_std=round(float(np.std(eye_vals)),   2),
            head_yaw_mean=round(float(np.mean(yaw_vals)),    2),
            head_yaw_std=round(float(np.std(yaw_vals)),      2),
            head_pitch_mean=round(float(np.mean(pitch_vals)),2),
            head_pitch_std=round(float(np.std(pitch_vals)),  2),
            smile_mean=round(float(np.mean(smile_vals)),     2),
            facial_movement_mean=round(float(np.mean(move_vals)), 3),
            posture_stability=round(float(np.mean(stable_vals)),  3),
            landmark_jitter_mean=round(float(np.mean(jitter_vals)), 3),
            face_presence_ratio=round(face_presence_ratio, 3),
        )

    def reset(self) -> None:
        """Clear all samples. Call at the start of a new interview session."""
        self._samples.clear()

    @property
    def sample_count(self) -> int:
        """Number of samples currently in the window."""
        return len(self._samples)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _prune(self, now: float) -> None:
        """Remove samples older than the configured window."""
        cutoff = now - self._window_sec
        while self._samples and self._samples[0].timestamp < cutoff:
            self._samples.popleft()
