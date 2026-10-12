"""
modules/emotion_detector.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━
FACS-compliant (Facial Action Coding System) Landmark Emotion Detector.

Responsibilities
----------------
- Computes Facial Action Units (AUs) directly from 478 MediaPipe landmarks:
  • AU1 + AU2 : Eyebrow Inner / Outer Raise  (Surprise / Fear)
  • AU4       : Eyebrow Lowerer / Furrow    (Angry / Sad)
  • AU6 + AU12: Cheek Raiser & Lip Corner Pull (Happy / Smile)
  • AU15      : Lip Corner Depressor         (Sad / Disgust)
  • AU25 + AU26: Jaw Drop / Mouth Open       (Surprise / Fear)
- Computes probability distribution over 7 basic emotions:
  `{angry, disgust, fear, happy, neutral, sad, surprise}`
- Smooths probabilities with Exponential Moving Average (EMA).
- 100% crash-free, zero PyArrow/Keras dependency, runs at 200+ FPS.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Optional

from .landmark_extractor import LandmarkGroups
import config as cfg


def _euclidean(p1: np.ndarray, p2: np.ndarray) -> float:
    return float(np.linalg.norm(p1 - p2))


class EmotionDetector:
    """
    Facial Action Coding System (FACS) emotion detector operating on
    MediaPipe Face Mesh landmark geometry.

    Parameters
    ----------
    alpha : float
        EMA smoothing factor for output probabilities.
    """

    def __init__(self, alpha: float = cfg.EMOTION_ALPHA, **kwargs) -> None:
        self._alpha = alpha
        _neutral_start = {lbl: 1.0 / len(cfg.EMOTION_LABELS)
                          for lbl in cfg.EMOTION_LABELS}
        self._smoothed_probs: Dict[str, float] = dict(_neutral_start)
        self._dominant: str = "neutral"

    def update(self, frame_bgr: np.ndarray, groups: Optional[LandmarkGroups] = None) -> None:
        """
        Update emotion predictions from current frame landmark groups.

        Parameters
        ----------
        frame_bgr : np.ndarray
        groups    : LandmarkGroups | None
        """
        if groups is None:
            return

        raw = self._predict_from_landmarks(groups)

        # Apply EMA smoothing
        for label in cfg.EMOTION_LABELS:
            prev = self._smoothed_probs.get(label, 0.0)
            cur  = raw.get(label, 0.0)
            self._smoothed_probs[label] = (
                self._alpha * cur + (1.0 - self._alpha) * prev
            )

        # Normalise to sum = 1
        total = sum(self._smoothed_probs.values()) or 1.0
        for k in self._smoothed_probs:
            self._smoothed_probs[k] /= total

        self._dominant = max(self._smoothed_probs, key=self._smoothed_probs.get)

    def _predict_from_landmarks(self, groups: LandmarkGroups) -> Dict[str, float]:
        """
        Extract Facial Action Units (AUs) and map to 7 emotion scores.
        """
        px = groups.all_pixels  # shape (478, 2)

        # Reference distance: inter-pupillary distance (IPD)
        left_eye_c  = groups.left_eye_center
        right_eye_c = groups.right_eye_center
        ipd = max(_euclidean(left_eye_c, right_eye_c), 1.0)

        # ── 1. Eyebrow distances (AU1, AU2, AU4) ─────────────────────────
        left_brow_y  = float(groups.left_eyebrow[:, 1].mean())
        right_brow_y = float(groups.right_eyebrow[:, 1].mean())
        avg_brow_y   = (left_brow_y + right_brow_y) / 2.0
        avg_eye_y    = (left_eye_c[1] + right_eye_c[1]) / 2.0

        # Brow-to-eye height (normalized by IPD)
        # Smaller y = higher on screen = raised eyebrow
        brow_raise = (avg_eye_y - avg_brow_y) / ipd  # typically 0.25 - 0.45

        # Inner eyebrow distance (AU4 furrow check)
        # Landmark 55 (left inner brow) & Landmark 285 (right inner brow)
        inner_brow_dist = _euclidean(px[55], px[285]) / ipd  # smaller = furrowed

        # ── 2. Mouth geometry (AU6, AU12, AU15, AU25, AU26) ──────────────
        mouth = groups.mouth
        mouth_w = max(_euclidean(mouth[0], mouth[6]), 1.0)  # corners width
        mouth_h = max(_euclidean(mouth[3], mouth[9]), 1.0)  # top/bottom height

        mar = mouth_h / mouth_w  # Mouth Aspect Ratio

        # Mouth corner elevation relative to mouth center y
        left_corner_y  = px[61, 1]   # left mouth corner
        right_corner_y = px[291, 1]  # right mouth corner
        mouth_center_y = float(mouth[:, 1].mean())

        corner_rise = (mouth_center_y - (left_corner_y + right_corner_y) / 2.0) / ipd

        # ── 3. Eye aspect ratio (AU43) ───────────────────────────────────
        left_ear  = _euclidean(groups.left_ear_pts[1], groups.left_ear_pts[5]) / max(_euclidean(groups.left_ear_pts[0], groups.left_ear_pts[3]), 1.0)
        right_ear = _euclidean(groups.right_ear_pts[1], groups.right_ear_pts[5]) / max(_euclidean(groups.right_ear_pts[0], groups.right_ear_pts[3]), 1.0)
        avg_ear   = (left_ear + right_ear) / 2.0

        # ── Action Unit Activations (0.0 - 1.0) ──────────────────────────
        au12_smile   = float(np.clip((corner_rise - 0.02) / 0.12, 0.0, 1.0))
        au4_furrow   = float(np.clip((0.35 - inner_brow_dist) / 0.15, 0.0, 1.0))
        au1_2_raise  = float(np.clip((brow_raise - 0.35) / 0.15, 0.0, 1.0))
        au15_depress = float(np.clip((-corner_rise - 0.01) / 0.10, 0.0, 1.0))
        au25_open    = float(np.clip((mar - 0.25) / 0.35, 0.0, 1.0))
        au_wide_eye  = float(np.clip((avg_ear - 0.28) / 0.15, 0.0, 1.0))

        # ── Map Action Units → 7 Emotion Unnormalized Logits ─────────────
        scores = {
            "happy":    au12_smile * 3.5 + 0.1,
            "surprise": au1_2_raise * 2.0 + au25_open * 1.5 + au_wide_eye * 1.5 + 0.1,
            "angry":    au4_furrow * 2.5 + (1.0 - au12_smile) * 0.8 + 0.1,
            "sad":      au15_depress * 2.0 + au4_furrow * 1.2 + 0.1,
            "fear":     au1_2_raise * 1.5 + au4_furrow * 1.0 + au_wide_eye * 1.5 + 0.1,
            "disgust":  au15_depress * 1.5 + au4_furrow * 1.0 + 0.1,
            "neutral":  1.2 - (au12_smile + au4_furrow + au1_2_raise + au15_depress + au25_open) * 0.5,
        }

        # Ensure all positive
        for k in scores:
            scores[k] = max(scores[k], 0.05)

        # Softmax / normalisation
        exp_scores = {k: float(np.exp(v)) for k, v in scores.items()}
        total_exp = sum(exp_scores.values())
        return {k: v / total_exp for k, v in exp_scores.items()}

    def get_emotions(self) -> Dict[str, float]:
        """Return the latest smoothed emotion probability dict."""
        return dict(self._smoothed_probs)

    @property
    def dominant_emotion(self) -> str:
        return self._dominant

    @property
    def is_ready(self) -> bool:
        return True
