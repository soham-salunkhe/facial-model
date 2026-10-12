"""
modules/smile_detector.py
━━━━━━━━━━━━━━━━━━━━━━━━━
Smile score computation from mouth landmarks.

Responsibilities
----------------
- Compute Mouth Aspect Ratio (MAR) to detect open-mouth events.
- Estimate smile score from the vertical rise of mouth corners
  relative to the center of the mouth.
- Maintain a running baseline to compensate for inter-subject differences.
"""

from __future__ import annotations

import numpy as np

from .landmark_extractor import LandmarkGroups
import config as cfg


class SmileResult:
    """Smile analysis output for one frame."""

    __slots__ = ("smile_score", "mouth_open_ratio", "is_smiling", "is_mouth_open")

    def __init__(
        self,
        smile_score: float,
        mouth_open_ratio: float,
        is_smiling: bool,
        is_mouth_open: bool,
    ) -> None:
        self.smile_score      = smile_score       # 0–100
        self.mouth_open_ratio = mouth_open_ratio  # 0–1
        self.is_smiling       = is_smiling
        self.is_mouth_open    = is_mouth_open


class SmileDetector:
    """
    Computes a smile score and mouth open ratio from mouth landmarks.

    The smile score is based on how much the mouth corners are raised
    relative to the midpoint of the mouth (vertical displacement normalized
    by mouth width). A small exponential moving average tracks the
    "neutral" baseline per-session to handle individual differences.

    Parameters
    ----------
    baseline_alpha : float
        EMA weight for the neutral baseline update (slow adaptation).
    """

    def __init__(self, baseline_alpha: float = 0.01) -> None:
        self._baseline_alpha = baseline_alpha
        self._corner_baseline: float | None = None  # neutral corner height
        self._mar_baseline:    float | None = None  # neutral MAR

    def analyze(self, groups: LandmarkGroups) -> SmileResult:
        """
        Compute smile metrics from mouth landmark groups.

        Parameters
        ----------
        groups : LandmarkGroups

        Returns
        -------
        SmileResult
        """
        mouth_pts = groups.mouth  # shape (12, 2)

        # ── Mouth geometry ───────────────────────────────────────────────
        # Horizontal span (left corner → right corner)
        x_coords = mouth_pts[:, 0]
        y_coords = mouth_pts[:, 1]
        mouth_width  = float(x_coords.max() - x_coords.min())
        mouth_height = float(y_coords.max() - y_coords.min())

        if mouth_width < 1.0:
            return SmileResult(0.0, 0.0, False, False)

        # Mouth Aspect Ratio
        mar = mouth_height / mouth_width

        # ── Smile estimation ─────────────────────────────────────────────
        # Identify left corner (min x), right corner (max x), mouth center y
        left_corner_idx  = int(np.argmin(mouth_pts[:, 0]))
        right_corner_idx = int(np.argmax(mouth_pts[:, 0]))

        left_corner_y  = mouth_pts[left_corner_idx, 1]
        right_corner_y = mouth_pts[right_corner_idx, 1]
        center_y       = float(y_coords.mean())

        # Corner raise: how much corners are above (lower y value) than center
        # Positive = corners are raised above center = smile
        corner_raise = center_y - (left_corner_y + right_corner_y) / 2.0

        # Normalise by mouth width
        corner_raise_norm = corner_raise / mouth_width

        # ── Baseline adaptation ──────────────────────────────────────────
        if self._corner_baseline is None:
            self._corner_baseline = float(np.clip(
                corner_raise_norm,
                cfg.SMILE_BASELINE_MIN,
                cfg.SMILE_BASELINE_MAX,
            ))
        else:
            a = self._baseline_alpha
            new_baseline = a * corner_raise_norm + (1.0 - a) * self._corner_baseline
            # Clamp to configured range to prevent drift during sustained smiling.
            # Without this, a continuously smiling candidate would eventually have
            # a baseline equal to their smile level, making their score drop to 0.
            self._corner_baseline = float(np.clip(
                new_baseline,
                cfg.SMILE_BASELINE_MIN,
                cfg.SMILE_BASELINE_MAX,
            ))

        if self._mar_baseline is None:
            self._mar_baseline = mar
        else:
            a = self._baseline_alpha
            self._mar_baseline = a * mar + (1.0 - a) * self._mar_baseline

        # ── Score computation ────────────────────────────────────────────
        # How much above baseline is the corner raise?
        delta = corner_raise_norm - self._corner_baseline
        # Scale: 0.1 normalised units ≈ 100% smile
        smile_score = float(np.clip(delta / 0.10 * 100.0, 0.0, 100.0))

        is_smiling    = smile_score > 40.0
        is_mouth_open = mar > cfg.MAR_OPEN_THRESHOLD

        return SmileResult(
            smile_score      = smile_score,
            mouth_open_ratio = float(np.clip(mar, 0.0, 1.0)),
            is_smiling       = is_smiling,
            is_mouth_open    = is_mouth_open,
        )

    def reset_baseline(self) -> None:
        """Reset the adaptive baseline (call between interview sessions)."""
        self._corner_baseline = None
        self._mar_baseline    = None
