"""
modules/pipeline.py
━━━━━━━━━━━━━━━━━━━
Encapsulated facial analysis pipeline for headless, server, and GUI operations.

Orchestrates all Phase 2 computer-vision components:
  FaceDetector → LandmarkExtractor → EyeContactAnalyzer → HeadPoseEstimator
  → SmileDetector → EmotionDetector → FeatureAggregator → InterviewClassifier
  → BehavioralWindowAggregator → build_analysis_frame()

Both main.py (OpenCV GUI) and server.py (FastAPI / WebSocket) rely on this
canonical pipeline implementation.
"""

from __future__ import annotations

import time
import uuid
from typing import Dict, NamedTuple, Optional, Tuple

import cv2
import numpy as np

import config as cfg
from .face_detector import (
    FaceDetector,
    FaceDetectionResult,
    FACE_STATUS_SINGLE,
    FACE_STATUS_MULTIPLE,
    FACE_STATUS_NO_FACE,
)
from .landmark_extractor import LandmarkExtractor, LandmarkGroups
from .eye_contact import EyeContactAnalyzer, EyeContactResult
from .head_pose import HeadPoseEstimator, HeadPoseResult
from .emotion_detector import EmotionDetector
from .smile_detector import SmileDetector, SmileResult
from .feature_aggregator import FeatureAggregator
from .interview_classifier import InterviewClassifier
from .behavioral_window import (
    BehavioralWindowAggregator,
    BehavioralWindowSnapshot,
)
from .analysis_output import build_analysis_frame


class PipelineFrameResult(NamedTuple):
    """Container for all results produced by processing a single video frame."""
    analysis: dict
    snapshot: Optional[BehavioralWindowSnapshot]
    should_emit_snapshot: bool
    detection: FaceDetectionResult
    groups: Optional[LandmarkGroups]
    eye_result: Optional[EyeContactResult]
    head_result: Optional[HeadPoseResult]
    smile_result: Optional[SmileResult]
    emotions: Dict[str, float]
    states: Dict[str, float]
    face_absent_frames: int
    was_face_present: bool


class FacialAnalysisPipeline:
    """
    Stateful execution pipeline for real-time facial behavioral analysis.
    """

    def __init__(
        self,
        auto_train_classifier: bool = True,
        snapshot_interval_sec: float = cfg.WS_SNAPSHOT_INTERVAL_SEC,
    ) -> None:
        self.snapshot_interval_sec = snapshot_interval_sec

        # ── Initialize core modules ──────────────────────────────────────
        self.face_detector      = FaceDetector()
        self.landmark_extractor = LandmarkExtractor()
        self.eye_analyzer       = EyeContactAnalyzer(fps=cfg.TARGET_FPS)
        self.head_estimator     = HeadPoseEstimator()
        self.emotion_detector   = EmotionDetector()
        self.smile_detector     = SmileDetector()
        self.feature_agg        = FeatureAggregator()
        self.classifier         = InterviewClassifier(auto_train=auto_train_classifier)
        self.behavioral_window  = BehavioralWindowAggregator(window_sec=cfg.BEHAVIORAL_WINDOW_SEC)

        # ── Session State ────────────────────────────────────────────────
        self.session_id: str           = str(uuid.uuid4())
        self.session_start_time: float = time.time()
        self._last_snapshot_mono: float = time.monotonic()
        self.total_frames: int         = 0
        self.face_absent_frames: int   = 0
        self.was_face_present: bool    = False

        # Current state and emotion priors
        self.emotions: Dict[str, float] = {lbl: 1.0 / len(cfg.EMOTION_LABELS) for lbl in cfg.EMOTION_LABELS}
        self.states: Dict[str, float]   = {lbl: 20.0 for lbl in cfg.STATE_LABELS}

    def process_frame(
        self,
        frame: np.ndarray,
        timestamp: Optional[float] = None,
    ) -> PipelineResult:
        """
        Process a single BGR image frame through the complete analysis pipeline.

        Parameters
        ----------
        frame     : BGR numpy array (H, W, 3).
        timestamp : Optional custom timestamp in seconds; defaults to time.time().

        Returns
        -------
        PipelineFrameResult containing structured JSON analysis dict, window snapshot,
        and intermediate module outputs.
        """
        self.total_frames += 1
        now_mono = time.monotonic()

        # ── Step 1: Face Detection ───────────────────────────────────────
        detection      = self.face_detector.detect(frame)
        face_status    = detection.status
        face_detected  = (face_status == FACE_STATUS_SINGLE)
        multiple_faces = (face_status == FACE_STATUS_MULTIPLE)

        groups: Optional[LandmarkGroups]      = None
        eye_result: Optional[EyeContactResult]  = None
        head_result: Optional[HeadPoseResult]   = None
        smile_result_val: Optional[SmileResult] = None
        feat_vec                              = None

        if face_detected:
            # ── Handle reappearance after hard absence ───────────────────
            if not self.was_face_present and self.face_absent_frames >= cfg.FACE_ABSENT_HARD_FRAMES:
                self.feature_agg.reset_temporal_state()

            self.face_absent_frames = 0
            self.was_face_present   = True
            face_result             = detection.face

            # ── Landmark Extraction ──────────────────────────────────────
            groups = self.landmark_extractor.extract(face_result)

            # ── Eye Contact & Blink ──────────────────────────────────────
            eye_result = self.eye_analyzer.analyze(groups)

            # ── Head Pose (EMA-smoothed) ─────────────────────────────────
            head_result = self.head_estimator.estimate(groups)

            # ── Smile Detection (baseline-clamped) ───────────────────────
            smile_result_val = self.smile_detector.analyze(groups)

            # ── Emotion Detection (FACS geometric) ───────────────────────
            self.emotion_detector.update(frame, groups)
            self.emotions = self.emotion_detector.get_emotions()

            # ── Feature Aggregation ──────────────────────────────────────
            feat_vec = self.feature_agg.build(
                groups, self.emotions, eye_result, head_result, smile_result_val
            )

            # ── Behavioral Pattern Classification ────────────────────────
            if self.classifier.is_ready:
                self.states = self.classifier.predict(feat_vec.array)

            # ── Behavioral Window Accumulation ───────────────────────────
            self.behavioral_window.add_frame(
                face_detected   = True,
                eye_contact_pct = float(eye_result.eye_contact_pct) if eye_result else 0.0,
                head_yaw        = float(head_result.yaw) if head_result else 0.0,
                head_pitch      = float(head_result.pitch) if head_result else 0.0,
                head_roll       = float(head_result.roll) if head_result else 0.0,
                smile_score     = float(smile_result_val.smile_score) if smile_result_val else 0.0,
                facial_movement = float(feat_vec.array[17]) if feat_vec else 0.0,
                is_head_stable  = bool(head_result.is_stable) if head_result else False,
                landmark_jitter = float(feat_vec.array[21]) if feat_vec else 0.0,
            )

        else:
            # ── No usable face (absent or multiple) ──────────────────────
            self.was_face_present = False
            self.face_absent_frames += 1

            self.emotion_detector.update(frame)
            self.emotions = self.emotion_detector.get_emotions()
            self.behavioral_window.add_frame(face_detected=False)

            if self.face_absent_frames >= cfg.FACE_ABSENT_HARD_FRAMES:
                self.states = {lbl: 20.0 for lbl in cfg.STATE_LABELS}

        # ── Build structured per-frame JSON dict ─────────────────────────
        analysis = build_analysis_frame(
            detection    = detection,
            eye_result   = eye_result,
            head_result  = head_result,
            smile_result = smile_result_val,
            emotions     = self.emotions,
            states       = self.states,
        )

        # ── Periodic snapshot check ──────────────────────────────────────
        should_emit_snapshot = False
        if (now_mono - self._last_snapshot_mono) >= self.snapshot_interval_sec:
            should_emit_snapshot = True
            self._last_snapshot_mono = now_mono

        current_snapshot = self.behavioral_window.get_snapshot()

        return PipelineFrameResult(
            analysis             = analysis,
            snapshot             = current_snapshot,
            should_emit_snapshot = should_emit_snapshot,
            detection            = detection,
            groups               = groups,
            eye_result           = eye_result,
            head_result          = head_result,
            smile_result         = smile_result_val,
            emotions             = self.emotions,
            states               = self.states,
            face_absent_frames   = self.face_absent_frames,
            was_face_present     = self.was_face_present,
        )

    def reset_session(self, new_session_id: bool = True) -> dict:
        """
        Reset all session metrics, counters, and baselines.

        Parameters
        ----------
        new_session_id : bool
            Whether to assign a fresh session UUID and restart the elapsed clock.
        """
        if new_session_id:
            self.session_id         = str(uuid.uuid4())
            self.session_start_time = time.time()
            self.total_frames       = 0

        self.eye_analyzer.reset()
        self.smile_detector.reset_baseline()
        self.feature_agg.reset_temporal_state()
        self.behavioral_window.reset()

        self.states             = {lbl: 20.0 for lbl in cfg.STATE_LABELS}
        self.emotions           = {lbl: 1.0 / len(cfg.EMOTION_LABELS) for lbl in cfg.EMOTION_LABELS}
        self.face_absent_frames = 0
        self.was_face_present   = False
        self._last_snapshot_mono = time.monotonic()

        return self.get_session_status()

    def get_session_status(self) -> dict:
        """Return the current session status and aggregated behavioral snapshot."""
        now = time.time()
        snapshot = self.behavioral_window.get_snapshot()
        snapshot_dict = snapshot._asdict() if snapshot is not None else None

        return {
            "session_id":          self.session_id,
            "is_active":           True,
            "started_at":          self.session_start_time,
            "elapsed_seconds":     round(max(0.0, now - self.session_start_time), 2),
            "total_frames":        self.total_frames,
            "face_absent_frames":  self.face_absent_frames,
            "is_face_present":     self.was_face_present,
            "behavioral_snapshot": snapshot_dict,
        }

    def close(self) -> None:
        """Release any underlying MediaPipe detector resources."""
        self.face_detector.close()
