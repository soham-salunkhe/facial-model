"""
modules/analysis_output.py
━━━━━━━━━━━━━━━━━━━━━━━━━━
Builds a structured, JSON-serializable per-frame analysis dictionary from
all module outputs.

Design principle
----------------
Every field is either:
  - Directly measured (landmark pixel position)
  - Geometrically derived (EAR, MAR, solvePnP angles)
  - FACS-inspired geometric estimate (emotion scores)
  - Synthetic-data-trained pattern estimate (behavioral_pattern)

No field claims to measure psychological state. Labels that involve
interpretation (e.g., "confident-like") are explicitly marked with
"-like" to indicate they are pattern-match estimates, not ground truth.

The classifier_note field in behavioral_pattern is always included and must
never be omitted in downstream applications.
"""

from __future__ import annotations

import time
from typing import Dict, Optional

from .face_detector import FaceDetectionResult, FACE_STATUS_SINGLE
from .eye_contact   import EyeContactResult
from .head_pose     import HeadPoseResult
from .smile_detector import SmileResult
import config as cfg


def build_analysis_frame(
    detection:      FaceDetectionResult,
    eye_result:     Optional[EyeContactResult],
    head_result:    Optional[HeadPoseResult],
    smile_result:   Optional[SmileResult],
    emotions:       Dict[str, float],
    states:         Dict[str, float],
) -> dict:
    """
    Build a structured JSON-serializable analysis dictionary for the current frame.

    Parameters
    ----------
    detection    : FaceDetectionResult from FaceDetector.detect()
    eye_result   : EyeContactResult or None (None when no face detected)
    head_result  : HeadPoseResult or None
    smile_result : SmileResult or None
    emotions     : Dict[str, float] from EmotionDetector.get_emotions()  (0–1 each)
    states       : Dict[str, float] from InterviewClassifier.predict()   (0–100 each)

    Returns
    -------
    dict
        JSON-serializable per-frame analysis. All values use the live
        estimates from this frame; no values are fabricated.
    """
    now = time.time()
    face_detected = detection.status == FACE_STATUS_SINGLE

    # ── Face status ─────────────────────────────────────────────────────────
    face_section = {
        "detected":   face_detected,
        "status":     detection.status,
        "face_count": detection.face_count,
    }

    # ── Eye contact ──────────────────────────────────────────────────────────
    if eye_result is not None:
        eye_section = {
            "percentage":           round(float(eye_result.eye_contact_pct), 1),
            "gaze":                 eye_result.gaze_direction.lower(),
            "blink_rate_per_min":   round(float(eye_result.blink_rate), 1),
            "blink_count":          int(eye_result.blink_count),
            "is_blinking":          bool(eye_result.is_blinking),
            "avg_ear":              round(float(eye_result.avg_ear), 3),
            # Signal type annotation
            "_signal_type":         "geometric_estimate",
        }
    else:
        eye_section = {
            "percentage": None, "gaze": None, "blink_rate_per_min": None,
            "blink_count": None, "is_blinking": None, "avg_ear": None,
            "_signal_type": "unavailable",
        }

    # ── Head pose ────────────────────────────────────────────────────────────
    if head_result is not None:
        head_section = {
            "pitch":       round(float(head_result.pitch), 2),
            "yaw":         round(float(head_result.yaw), 2),
            "roll":        round(float(head_result.roll), 2),
            "orientation": head_result.label,
            "is_stable":   bool(head_result.is_stable),
            # Raw unsmoothed angles for debugging
            "raw_pitch":   round(float(head_result.raw_pitch), 2),
            "raw_yaw":     round(float(head_result.raw_yaw), 2),
            "raw_roll":    round(float(head_result.raw_roll), 2),
            "_signal_type": "geometric_estimate_uncalibrated_camera",
        }
    else:
        head_section = {
            "pitch": None, "yaw": None, "roll": None,
            "orientation": None, "is_stable": None,
            "raw_pitch": None, "raw_yaw": None, "raw_roll": None,
            "_signal_type": "unavailable",
        }

    # ── Facial expression ────────────────────────────────────────────────────
    # IMPORTANT: These are FACS-inspired geometric estimates, NOT validated
    # emotion probabilities. The weights are manually crafted, not learned
    # from real data. Label them clearly as estimates.
    dominant_expression = "neutral"
    if emotions:
        dominant_expression = max(emotions, key=emotions.get)

    expression_section = {
        "smile_score":      round(float(smile_result.smile_score), 1)    if smile_result else None,
        "mouth_open":       bool(smile_result.is_mouth_open)              if smile_result else None,
        "expression_estimate": dominant_expression,
        "expression_probabilities": {
            k: round(float(v), 4) for k, v in emotions.items()
        } if emotions else {},
        # Always include methodology note — never omit
        "_signal_type": "FACS_inspired_geometric_heuristic",
        "_note": (
            "Expression probabilities are derived from facial geometry (Action Units). "
            "They are NOT validated emotion scores. Do not use for psychological assessment."
        ),
    }

    # ── Behavioral pattern ───────────────────────────────────────────────────
    # The classifier was trained on SYNTHETIC data only. Its output should be
    # read as "resemblance to a behavioral archetype", not as psychological state.
    if states:
        dominant_state = max(states, key=states.get)
        pattern_section = {
            "dominant_pattern": f"{dominant_state.lower()}-like",
            "probabilities": {
                f"{k.lower()}-like": round(float(v) / 100.0, 4)
                for k, v in states.items()
            },
            "_signal_type": "synthetic_data_trained_pattern_estimate",
            "_note": (
                "Output from an XGBoost classifier trained on synthetic data only. "
                "NOT validated on real interview recordings. "
                "Interpret as pattern resemblance, not psychological ground truth."
            ),
        }
    else:
        pattern_section = {
            "dominant_pattern": None,
            "probabilities": {},
            "_signal_type": "synthetic_data_trained_pattern_estimate",
            "_note": "Classifier not ready.",
        }

    return {
        "timestamp":          round(now, 3),
        "face":               face_section,
        "eye_contact":        eye_section,
        "head_pose":          head_section,
        "facial_expression":  expression_section,
        "behavioral_pattern": pattern_section,
    }
