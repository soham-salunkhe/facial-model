"""
modules/feature_aggregator.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Combines all modular analysis results into a single 22-D feature vector
suitable for the interview state classifier.

Feature Vector Layout (22 dimensions)
---------------------------------------
Index  Feature
  0    Emotion: angry        (0–1)
  1    Emotion: disgust      (0–1)
  2    Emotion: fear         (0–1)
  3    Emotion: happy        (0–1)
  4    Emotion: neutral      (0–1)
  5    Emotion: sad          (0–1)
  6    Emotion: surprise     (0–1)
  7    Eye contact %         (0–1, normalised from 0–100)
  8    Blink rate / 30       (normalised; 30 blinks/min is typical nervous rate)
  9    Gaze left flag        (0/1)
 10    Gaze right flag       (0/1)
 11    Gaze up flag          (0/1)
 12    Gaze down flag        (0/1)
 13    Head pitch / 90       (normalised)
 14    Head yaw / 90         (normalised)
 15    Head roll / 45        (normalised)
 16    Smile score / 100     (0–1)
 17    Facial movement delta (0–1, clamped)
 18    Mouth open ratio      (0–1)
 19    Eyebrow raise score   (0–1)
 20    Head pose stability   (1 = stable, 0 = not)
 21    Landmark jitter       (0–1, clamped std-dev of motion)
"""

from __future__ import annotations

from typing import Dict, NamedTuple, Optional

import numpy as np

from .eye_contact        import EyeContactResult
from .head_pose          import HeadPoseResult
from .landmark_extractor import LandmarkGroups
from .smile_detector     import SmileResult
import config as cfg


class FeatureVector(NamedTuple):
    """Typed wrapper around the raw numpy feature array."""
    array:    np.ndarray   # shape (22,) float32
    metadata: Dict         # human-readable dict for debugging/display


class FeatureAggregator:
    """
    Aggregates per-frame analysis outputs into a 22-D feature vector.

    Also computes:
    - Facial movement delta (frame-to-frame landmark motion).
    - Eyebrow raise score (relative to eye position).
    - Landmark jitter (temporal std-dev of all landmark positions).

    Parameters
    ----------
    motion_alpha : float
        EMA factor for smoothing landmark motion delta.
    jitter_window : int
        Number of frames to compute jitter over.
    """

    def __init__(
        self,
        motion_alpha: float = 0.3,
        jitter_window: int = 10,
    ) -> None:
        self._motion_alpha = motion_alpha
        self._jitter_window = jitter_window

        self._prev_pixels: Optional[np.ndarray] = None
        self._smoothed_delta: float = 0.0
        self._pixel_history: list[np.ndarray] = []

    def build(
        self,
        groups: LandmarkGroups,
        emotions: Dict[str, float],
        eye_result: EyeContactResult,
        head_result: Optional[HeadPoseResult],
        smile_result: SmileResult,
    ) -> FeatureVector:
        """
        Build the 22-D feature vector from all module outputs.

        Parameters
        ----------
        groups      : LandmarkGroups
        emotions    : Dict[str, float]  — from EmotionDetector.get_emotions()
        eye_result  : EyeContactResult  — from EyeContactAnalyzer.analyze()
        head_result : HeadPoseResult | None
        smile_result: SmileResult

        Returns
        -------
        FeatureVector
        """
        feat = np.zeros(cfg.FEATURE_DIM, dtype=np.float32)

        # ── [0–6] Emotion probabilities ──────────────────────────────────
        for i, label in enumerate(cfg.EMOTION_LABELS):
            feat[i] = float(emotions.get(label, 0.0))

        # ── [7] Eye contact % ────────────────────────────────────────────
        feat[7] = float(eye_result.eye_contact_pct) / 100.0

        # ── [8] Blink rate (normalised to 30 bpm) ───────────────────────
        feat[8] = float(np.clip(eye_result.blink_rate / 30.0, 0.0, 3.0))

        # ── [9–12] Gaze flags ────────────────────────────────────────────
        feat[9]  = float(eye_result.gaze_left)
        feat[10] = float(eye_result.gaze_right)
        feat[11] = float(eye_result.gaze_up)
        feat[12] = float(eye_result.gaze_down)

        # ── [13–15] Head pose angles (normalised) ────────────────────────
        if head_result is not None:
            feat[13] = float(np.clip(head_result.pitch / 90.0, -1.0, 1.0))
            feat[14] = float(np.clip(head_result.yaw   / 90.0, -1.0, 1.0))
            feat[15] = float(np.clip(head_result.roll  / 45.0, -1.0, 1.0))
        # else: leave as 0 (stable assumption)

        # ── [16] Smile score ─────────────────────────────────────────────
        feat[16] = float(smile_result.smile_score) / 100.0

        # ── [17] Facial movement delta ───────────────────────────────────
        current_pixels = groups.all_pixels  # shape (N, 2)
        if self._prev_pixels is not None and self._prev_pixels.shape == current_pixels.shape:
            delta = float(np.mean(np.linalg.norm(
                current_pixels - self._prev_pixels, axis=1
            )))
            self._smoothed_delta = (
                self._motion_alpha * delta
                + (1.0 - self._motion_alpha) * self._smoothed_delta
            )
        self._prev_pixels = current_pixels.copy()
        # Normalise: 5px average motion ≈ 1.0
        feat[17] = float(np.clip(self._smoothed_delta / 5.0, 0.0, 3.0))

        # ── [18] Mouth open ratio ─────────────────────────────────────────
        feat[18] = float(np.clip(smile_result.mouth_open_ratio, 0.0, 1.0))

        # ── [19] Eyebrow raise score ─────────────────────────────────────
        feat[19] = self._eyebrow_raise(groups)

        # ── [20] Head pose stability ──────────────────────────────────────
        feat[20] = 1.0 if (head_result is not None and head_result.is_stable) else 0.0

        # ── [21] Landmark jitter ──────────────────────────────────────────
        self._pixel_history.append(current_pixels.copy())
        if len(self._pixel_history) > self._jitter_window:
            self._pixel_history.pop(0)
        feat[21] = self._compute_jitter()

        # ── Build metadata dict ───────────────────────────────────────────
        meta = {
            "emotions":          {k: round(float(emotions.get(k, 0.0)), 3)
                                  for k in cfg.EMOTION_LABELS},
            "eye_contact_pct":   round(float(eye_result.eye_contact_pct), 1),
            "blink_rate":        round(float(eye_result.blink_rate), 1),
            "gaze_direction":    eye_result.gaze_direction,
            "head_pose":         head_result.label if head_result else "Unknown",
            "smile_score":       round(float(smile_result.smile_score), 1),
            "mouth_open_ratio":  round(float(smile_result.mouth_open_ratio), 3),
            "movement_delta":    round(float(self._smoothed_delta), 3),
            "eyebrow_raise":     round(float(feat[19]), 3),
            "jitter":            round(float(feat[21]), 3),
        }

        return FeatureVector(array=feat, metadata=meta)

    def _eyebrow_raise(self, groups: LandmarkGroups) -> float:
        """
        Compute a 0–1 eyebrow raise score.

        Measures vertical distance from eyebrow center to eye center,
        normalised by the inter-eye distance.
        """
        left_brow_y  = float(groups.left_eyebrow[:, 1].mean())
        right_brow_y = float(groups.right_eyebrow[:, 1].mean())
        left_eye_y   = float(groups.left_eye[:, 1].mean())
        right_eye_y  = float(groups.right_eye[:, 1].mean())

        # Brow is above eye → smaller y value (image coords: y increases downward)
        brow_eye_gap_left  = left_eye_y  - left_brow_y
        brow_eye_gap_right = right_eye_y - right_brow_y
        avg_gap = (brow_eye_gap_left + brow_eye_gap_right) / 2.0

        # Normalise by inter-eye horizontal distance
        left_eye_x  = float(groups.left_eye[:, 0].mean())
        right_eye_x = float(groups.right_eye[:, 0].mean())
        inter_eye = max(abs(right_eye_x - left_eye_x), 1.0)

        score = avg_gap / inter_eye  # typically 0.15–0.45
        # Remap to 0–1: 0.10 = neutral, 0.40 = very raised
        score = float(np.clip((score - 0.10) / 0.30, 0.0, 1.0))
        return score

    def _compute_jitter(self) -> float:
        """Compute temporal jitter as mean std-dev of landmark positions."""
        if len(self._pixel_history) < 2:
            return 0.0
        stack = np.stack(self._pixel_history, axis=0)  # (T, N, 2)
        std_per_landmark = np.std(stack, axis=0)        # (N, 2)
        jitter = float(np.mean(std_per_landmark))
        return float(np.clip(jitter / 5.0, 0.0, 1.0))  # normalise

    def reset_temporal_state(self) -> None:
        """
        Reset all frame-to-frame temporal state.

        Call this when:
        - The face has been absent for a significant number of frames and then
          reappears, to prevent comparing the new position against the old one.
        - A new interview session begins.

        Without this, the first frame after a long absence would calculate a
        large movement delta (distance from pre-absence to post-absence position)
        and a jitter spike, corrupting features [17] and [21].
        """
        self._prev_pixels    = None
        self._smoothed_delta = 0.0
        self._pixel_history  = []

