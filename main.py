"""
main.py — Entry point for the AI-powered Facial Analysis Module.

Phase 2 changes
---------------
- FaceDetector.detect() now returns FaceDetectionResult (status + optional face).
- Explicit handling for: single_face, no_face, multiple_faces.
- Face absence tracking:
    face_absent_frames counts consecutive frames without a single face.
    < FACE_ABSENT_SOFT_FRAMES : normal (retain last values, no warning)
    >= FACE_ABSENT_SOFT_FRAMES : soft warning displayed
    >= FACE_ABSENT_HARD_FRAMES : hard absent — stale state probs reset to flat prior
- On face reappearance after hard absence: feature_agg.reset_temporal_state()
  is called to prevent the artificial movement spike from the position gap.
- BehavioralWindowAggregator accumulates per-frame signals.
- build_analysis_frame() produces structured JSON-ready output each frame.
- Session reset [R] now also resets behavioral window and temporal state.

Usage
-----
    python main.py               # Start with webcam 0
    python main.py --cam 1       # Use webcam index 1
    python main.py --width 1280  # Override capture width
    python main.py --height 720  # Override capture height
    python main.py --no-train    # Skip auto-training (requires existing model)
    python main.py --retrain     # Force retrain even if model exists

Controls (while running)
------------------------
    Q   — Quit
    R   — Reset session (all counters, baselines, window aggregation)
    S   — Save screenshot to screenshots/

Pipeline (per frame)
--------------------
  1. Capture frame from webcam
  2. FaceDetector → FaceDetectionResult  (no_face | single_face | multiple_faces)
  3. If single_face:
     3a. LandmarkExtractor → LandmarkGroups
     3b. EyeContactAnalyzer → EyeContactResult
     3c. HeadPoseEstimator → HeadPoseResult  (EMA-smoothed pitch/yaw/roll)
     3d. SmileDetector → SmileResult  (baseline-clamped)
     3e. EmotionDetector (FACS geometric) → emotions dict
     3f. FeatureAggregator → 22-D feature vector
     3g. InterviewClassifier → behavioral pattern probabilities
     3h. BehavioralWindowAggregator.add_frame()
  4. build_analysis_frame() → structured dict  (→ WebSocket in Phase 3)
  5. LiveDashboard.render() → composed HUD frame
  6. cv2.imshow()
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules import (
    FaceDetector,
    LandmarkExtractor,
    EyeContactAnalyzer,
    HeadPoseEstimator,
    EmotionDetector,
    SmileDetector,
    FeatureAggregator,
    InterviewClassifier,
    BehavioralWindowAggregator,
    FACE_STATUS_SINGLE,
    FACE_STATUS_MULTIPLE,
    FACE_STATUS_NO_FACE,
)
from modules.analysis_output import build_analysis_frame
from dashboard import LiveDashboard
import config as cfg


# ── CLI arguments ─────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AI-powered Facial Analysis Module — Mock Interview System"
    )
    p.add_argument("--cam",      type=int,  default=0,             help="Webcam index")
    p.add_argument("--width",    type=int,  default=cfg.CAM_WIDTH,  help="Capture width")
    p.add_argument("--height",   type=int,  default=cfg.CAM_HEIGHT, help="Capture height")
    p.add_argument("--no-train", action="store_true", help="Disable auto-training")
    p.add_argument("--retrain",  action="store_true", help="Force retrain model")
    return p.parse_args()


# ── Main loop ─────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()

    print("=" * 62)
    print("  AI Facial Analysis Module — Mock Interview System")
    print("=" * 62)

    # ── Optional retrain ─────────────────────────────────────────────────
    if args.retrain:
        import pickle
        for path in [cfg.CLASSIFIER_PATH, cfg.SCALER_PATH]:
            if os.path.exists(path):
                os.remove(path)
        print("[main] Model cache cleared — will retrain.")

    # ── Initialise all modules ───────────────────────────────────────────
    print("[main] Initialising modules…")
    face_detector      = FaceDetector()
    landmark_extractor = LandmarkExtractor()
    eye_analyzer       = EyeContactAnalyzer(fps=cfg.TARGET_FPS)
    head_estimator     = HeadPoseEstimator()
    emotion_detector   = EmotionDetector()
    smile_detector     = SmileDetector()
    feature_agg        = FeatureAggregator()
    classifier         = InterviewClassifier(auto_train=not args.no_train)
    dashboard          = LiveDashboard(sidebar_width=cfg.SIDEBAR_WIDTH)
    behavioral_window  = BehavioralWindowAggregator(window_sec=cfg.BEHAVIORAL_WINDOW_SEC)

    # Default states (flat prior — no information yet)
    emotions: dict = {lbl: 1.0 / 7 for lbl in cfg.EMOTION_LABELS}
    states:   dict = {lbl: 20.0    for lbl in cfg.STATE_LABELS}

    # ── Face absence tracking ────────────────────────────────────────────
    # face_absent_frames counts consecutive frames without a single usable face.
    # This is used to:
    #   - display soft/hard absence warnings in the HUD
    #   - reset stale state probabilities after FACE_ABSENT_HARD_FRAMES
    #   - trigger feature_agg.reset_temporal_state() on reappearance
    face_absent_frames: int  = 0
    was_face_present:   bool = False

    # ── Open webcam ──────────────────────────────────────────────────────
    print(f"[main] Opening webcam {args.cam}…")
    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print(f"[ERROR] Could not open webcam {args.cam}. Try --cam <index>.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
    cap.set(cv2.CAP_PROP_FPS,          cfg.TARGET_FPS)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[main] Capture resolution: {actual_w}×{actual_h}")
    print(f"[main] Dashboard window:   {actual_w + cfg.SIDEBAR_WIDTH}×{actual_h}")
    print(f"[main] Press [Q] to quit,  [R] to reset session,  [S] to screenshot.")
    print("=" * 62)

    # ── Screenshot directory ─────────────────────────────────────────────
    screenshot_dir = os.path.join(os.path.dirname(__file__), "screenshots")
    os.makedirs(screenshot_dir, exist_ok=True)
    screenshot_count = 0

    cv2.namedWindow(cfg.WINDOW_TITLE, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(cfg.WINDOW_TITLE, actual_w + cfg.SIDEBAR_WIDTH, actual_h)

    # ── Main capture loop ────────────────────────────────────────────────
    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[WARN] Empty frame — skipping.")
                continue

            frame_idx += 1

            # ── Face detection ──────────────────────────────────────────
            detection      = face_detector.detect(frame)
            face_status    = detection.status
            face_detected  = face_status == FACE_STATUS_SINGLE
            multiple_faces = face_status == FACE_STATUS_MULTIPLE

            # Per-frame results (reset each frame)
            groups           = None
            eye_result       = None
            head_result      = None
            smile_result_val = None
            feat_vec         = None

            if face_detected:
                # ── Handle reappearance after long absence ──────────────
                if not was_face_present and face_absent_frames >= cfg.FACE_ABSENT_HARD_FRAMES:
                    print(
                        f"[main] Face reappeared after {face_absent_frames} frame absence — "
                        "resetting temporal motion state."
                    )
                    feature_agg.reset_temporal_state()

                face_absent_frames = 0
                was_face_present   = True

                face_result = detection.face  # FaceResult (guaranteed non-None here)

                # ── Landmark extraction ─────────────────────────────────
                groups = landmark_extractor.extract(face_result)

                # ── Eye contact & blink ─────────────────────────────────
                eye_result = eye_analyzer.analyze(groups)

                # ── Head pose (EMA-smoothed) ────────────────────────────
                head_result = head_estimator.estimate(groups)

                # ── Smile detection (baseline-clamped) ──────────────────
                smile_result_val = smile_detector.analyze(groups)

                # ── Emotion detection (FACS geometric) ──────────────────
                emotion_detector.update(frame, groups)
                emotions = emotion_detector.get_emotions()

                # ── Feature aggregation ─────────────────────────────────
                feat_vec = feature_agg.build(
                    groups, emotions, eye_result,
                    head_result, smile_result_val
                )

                # ── Behavioral pattern classification ───────────────────
                if classifier.is_ready:
                    states = classifier.predict(feat_vec.array)

                # ── Behavioral window ───────────────────────────────────
                behavioral_window.add_frame(
                    face_detected    = True,
                    eye_contact_pct  = float(eye_result.eye_contact_pct),
                    head_yaw         = float(head_result.yaw)   if head_result else 0.0,
                    head_pitch       = float(head_result.pitch) if head_result else 0.0,
                    head_roll        = float(head_result.roll)  if head_result else 0.0,
                    smile_score      = float(smile_result_val.smile_score) if smile_result_val else 0.0,
                    facial_movement  = float(feat_vec.array[17]),
                    is_head_stable   = bool(head_result.is_stable) if head_result else False,
                    landmark_jitter  = float(feat_vec.array[21]),
                )

            else:
                # ── No usable face (absent or multiple) ─────────────────
                was_face_present    = False
                face_absent_frames += 1

                # Keep emotion thread alive (for quick re-detection)
                emotion_detector.update(frame)
                emotions = emotion_detector.get_emotions()

                # Record face-absent frame in behavioral window
                behavioral_window.add_frame(face_detected=False)

                # Hard absence: reset state probs to flat prior
                # to avoid showing stale "87% Confident-like" from minutes ago
                if face_absent_frames >= cfg.FACE_ABSENT_HARD_FRAMES:
                    states = {lbl: 20.0 for lbl in cfg.STATE_LABELS}

                if multiple_faces and (face_absent_frames == 1 or face_absent_frames % 60 == 0):
                    print(
                        f"[WARN] {detection.face_count} faces detected — "
                        "frame skipped (single-candidate mode only)."
                    )

            # ── Structured analysis output ──────────────────────────────
            # In Phase 3 this dict will be emitted over a WebSocket.
            # For now it's built each frame but only used locally.
            analysis = build_analysis_frame(
                detection    = detection,
                eye_result   = eye_result,
                head_result  = head_result,
                smile_result = smile_result_val,
                emotions     = emotions,
                states       = states,
            )
            # Uncomment to debug structured output:
            # import json; print(json.dumps(analysis, indent=2))

            # ── Render HUD ──────────────────────────────────────────────
            composed = dashboard.render(
                frame_bgr          = frame,
                groups             = groups,
                eye_result         = eye_result,
                head_result        = head_result,
                smile_result       = smile_result_val,
                emotions           = emotions,
                states             = states,
                face_detected      = face_detected,
                multiple_faces     = multiple_faces,
                face_absent_frames = face_absent_frames,
            )

            cv2.imshow(cfg.WINDOW_TITLE, composed)

            # ── Key handling ────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == ord("Q"):
                print("[main] Quit requested.")
                break

            elif key == ord("r") or key == ord("R"):
                print("[main] Resetting session…")
                eye_analyzer.reset()
                smile_detector.reset_baseline()
                feature_agg.reset_temporal_state()
                behavioral_window.reset()
                states             = {lbl: 20.0 for lbl in cfg.STATE_LABELS}
                face_absent_frames = 0
                was_face_present   = False

            elif key == ord("s") or key == ord("S"):
                screenshot_count += 1
                fname = os.path.join(
                    screenshot_dir,
                    f"screenshot_{screenshot_count:04d}.png"
                )
                cv2.imwrite(fname, composed)
                print(f"[main] Screenshot saved → {fname}")

            # Auto-quit if window is closed
            if cv2.getWindowProperty(cfg.WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1:
                break

    except KeyboardInterrupt:
        print("\n[main] Interrupted.")
    finally:
        cap.release()
        face_detector.close()
        cv2.destroyAllWindows()
        print("[main] Resources released. Goodbye.")


if __name__ == "__main__":
    main()
