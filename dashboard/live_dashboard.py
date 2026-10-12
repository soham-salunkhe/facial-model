"""
dashboard/live_dashboard.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━
Custom OpenCV real-time HUD for the Facial Analysis module.

Phase 2 changes
---------------
- "INTERVIEW STATES"  → "BEHAVIORAL PATTERN ESTIMATE"
- "Emotion:"          → "Facial Expression (est.)"
- State label suffix  "-like" appended (e.g. "Confident-like  62%")
- Multiple-face warning displayed prominently in sidebar and frame overlay
- Face-absent display: shows "Face Not Detected" after FACE_ABSENT_SOFT_FRAMES
- Hard-absent indicator after FACE_ABSENT_HARD_FRAMES

Layout
------
  ┌────────────────────────────────┬──────────────────────┐
  │  Webcam feed                   │  FACIAL ANALYSIS HUD  │
  │  + Face mesh overlay           │  ─────────────────── │
  │  + Head pose axis arrows       │  Face Detected  ✓    │
  │  + Gaze direction indicator    │  Expression (est.)    │
  │  + Blink flash ring            │  Eye Contact: 78%    │
  │                                │  Blink Rate: 14/min  │
  │                                │  Head Pose: Stable   │
  │                                │  Smile:  ████░░ 42%  │
  │                                │  ─────────────────── │
  │                                │  BEHAVIORAL PATTERN   │
  │                                │  ESTIMATE             │
  │                                │  Confident-like ████  │
  │                                │  ─────────────────── │
  │                                │  FPS: 28  [Q] Quit   │
  └────────────────────────────────┴──────────────────────┘
"""

from __future__ import annotations

import time
from typing import Dict, Optional

import cv2
import numpy as np

from modules.landmark_extractor import LandmarkGroups
from modules.eye_contact        import EyeContactResult
from modules.head_pose          import HeadPoseResult
from modules.smile_detector     import SmileResult
import config as cfg


# ── Typography helpers ────────────────────────────────────────────────────────

_FONT      = cv2.FONT_HERSHEY_SIMPLEX
_FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX

def _text(
    img: np.ndarray,
    msg: str,
    pos: tuple,
    scale: float = 0.55,
    color: tuple = cfg.COLOR_TEXT,
    thickness: int = 1,
    bold: bool = False,
) -> None:
    font = _FONT_BOLD if bold else _FONT
    cv2.putText(img, msg, pos, font, scale, color, thickness, cv2.LINE_AA)


def _bar(
    img: np.ndarray,
    x: int, y: int,
    width: int, height: int,
    pct: float,
    fill_color: tuple,
    label: str = "",
    show_pct: bool = True,
) -> None:
    """Draw a labelled progress bar."""
    cv2.rectangle(img, (x, y), (x + width, y + height), cfg.COLOR_BAR_BG, -1)
    fill_w = int(width * min(max(pct, 0.0), 100.0) / 100.0)
    if fill_w > 0:
        cv2.rectangle(img, (x, y), (x + fill_w, y + height), fill_color, -1)
    cv2.rectangle(img, (x, y), (x + width, y + height), (80, 80, 90), 1)
    if label:
        _text(img, label, (x - 2, y + height - 2), scale=0.42, color=cfg.COLOR_TEXT)
    if show_pct:
        _text(img, f"{pct:.0f}%", (x + width + 5, y + height - 2),
              scale=0.42, color=cfg.COLOR_SUBTEXT)


def _divider(img: np.ndarray, x: int, y: int, w: int) -> None:
    cv2.line(img, (x, y), (x + w, y), (60, 60, 75), 1)


# ── Main dashboard class ──────────────────────────────────────────────────────

class LiveDashboard:
    """
    Renders the full real-time interview analysis HUD.

    Call render() each frame to get the composed output image.

    Parameters
    ----------
    sidebar_width : int
        Width of the right-hand metrics panel in pixels.
    """

    def __init__(self, sidebar_width: int = cfg.SIDEBAR_WIDTH) -> None:
        self._sw = sidebar_width
        self._fps_history: list = []
        self._last_ts: float = time.monotonic()
        self._blink_flash_frames = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def render(
        self,
        frame_bgr:          np.ndarray,
        groups:             Optional[LandmarkGroups],
        eye_result:         Optional[EyeContactResult],
        head_result:        Optional[HeadPoseResult],
        smile_result:       Optional[SmileResult],
        emotions:           Dict[str, float],
        states:             Dict[str, float],
        face_detected:      bool,
        multiple_faces:     bool = False,
        face_absent_frames: int = 0,
    ) -> np.ndarray:
        """
        Compose and return the full HUD frame.

        Parameters
        ----------
        frame_bgr           : raw webcam frame
        groups              : LandmarkGroups | None
        eye_result          : EyeContactResult | None
        head_result         : HeadPoseResult | None
        smile_result        : SmileResult | None
        emotions            : smoothed expression probabilities (0–1)
        states              : behavioral pattern probabilities (0–100)
        face_detected       : True when exactly one face is detected
        multiple_faces      : True when >1 faces detected
        face_absent_frames  : consecutive frames without a single face
        """
        h, w = frame_bgr.shape[:2]

        # ── FPS ───────────────────────────────────────────────────────────
        now = time.monotonic()
        elapsed = now - self._last_ts
        self._last_ts = now
        if elapsed > 0:
            self._fps_history.append(1.0 / elapsed)
            if len(self._fps_history) > 30:
                self._fps_history.pop(0)
        fps_avg = float(np.mean(self._fps_history)) if self._fps_history else 0.0

        # ── Draw overlays on frame ─────────────────────────────────────────
        frame = frame_bgr.copy()
        if face_detected and groups is not None:
            self._draw_face_mesh(frame, groups)
            if head_result is not None:
                self._draw_head_pose_axis(frame, groups, head_result)
            if eye_result is not None:
                self._draw_gaze_indicator(frame, groups, eye_result)
                if eye_result.is_blinking:
                    self._blink_flash_frames = 4
            if self._blink_flash_frames > 0:
                self._draw_blink_flash(frame, groups)
                self._blink_flash_frames -= 1

        if multiple_faces:
            self._draw_multiple_face_warning(frame, w, h)

        if face_absent_frames >= cfg.FACE_ABSENT_HARD_FRAMES:
            self._draw_face_absent_overlay(frame, w, h)

        # ── Sidebar ───────────────────────────────────────────────────────
        sidebar = self._build_sidebar(
            h, face_detected, multiple_faces, face_absent_frames,
            eye_result, head_result, smile_result, emotions, states, fps_avg
        )

        # ── Composite ─────────────────────────────────────────────────────
        composed = np.zeros((h, w + self._sw, 3), dtype=np.uint8)
        composed[:h, :w] = cv2.resize(frame, (w, h))
        composed[:h, w:w + self._sw] = sidebar
        cv2.line(composed, (w, 0), (w, h), (60, 60, 75), 2)

        return composed

    # ── Webcam overlay drawing ────────────────────────────────────────────────

    def _draw_face_mesh(self, img: np.ndarray, groups: LandmarkGroups) -> None:
        """Draw a subset of face mesh connections."""
        for eye_pts in [groups.left_eye, groups.right_eye]:
            cv2.polylines(img, [eye_pts.astype(np.int32)],
                          True, cfg.COLOR_MESH, 1, cv2.LINE_AA)
        for iris_pts in [groups.left_iris, groups.right_iris]:
            c = iris_pts.mean(axis=0).astype(np.int32)
            r = int(np.linalg.norm(iris_pts[0] - iris_pts[2]) / 2)
            cv2.circle(img, tuple(c), max(r, 2), cfg.COLOR_IRIS, 1, cv2.LINE_AA)
        for brow_pts in [groups.left_eyebrow, groups.right_eyebrow]:
            cv2.polylines(img, [brow_pts.astype(np.int32)],
                          False, (180, 220, 100), 1, cv2.LINE_AA)
        cv2.polylines(img, [groups.mouth.astype(np.int32)],
                      True, (100, 180, 220), 1, cv2.LINE_AA)
        cv2.polylines(img, [groups.jawline.astype(np.int32)],
                      False, (80, 160, 200), 1, cv2.LINE_AA)
        nt = tuple(groups.nose_tip.astype(np.int32))
        cv2.circle(img, nt, 3, cfg.COLOR_MESH, -1, cv2.LINE_AA)

    def _draw_head_pose_axis(
        self, img: np.ndarray, groups: LandmarkGroups, head: HeadPoseResult
    ) -> None:
        """Draw 3D orientation axes from the nose tip."""
        axis_len = 60.0
        axis_3d = np.float32([
            [axis_len, 0,         0],
            [0,       -axis_len,  0],
            [0,        0,        -axis_len],
        ])
        nose_2d   = groups.nose_tip.astype(np.float32).reshape(1, 2)
        projected, _ = cv2.projectPoints(
            axis_3d, head.rvec, head.tvec, head.camera_matrix, np.zeros((4, 1))
        )
        origin = tuple(nose_2d[0].astype(np.int32))
        for i, color in enumerate([(0, 0, 220), (0, 220, 0), (220, 0, 0)]):
            end = tuple(projected[i, 0].astype(np.int32))
            cv2.arrowedLine(img, origin, end, color, 2, cv2.LINE_AA, tipLength=0.3)

    def _draw_gaze_indicator(
        self, img: np.ndarray, groups: LandmarkGroups, eye: EyeContactResult
    ) -> None:
        """Draw a gaze direction arrow between the eyes."""
        left_c  = groups.left_iris_center.astype(np.int32)
        right_c = groups.right_iris_center.astype(np.int32)
        mid     = (left_c + right_c) // 2
        offsets = {"Left": (-30, 0), "Right": (30, 0),
                   "Up": (0, -25), "Down": (0, 25), "Center": (0, 0)}
        offset = offsets.get(eye.gaze_direction, (0, 0))
        if offset != (0, 0):
            end   = (mid[0] + offset[0], mid[1] + offset[1])
            color = (80, 200, 255)
            cv2.arrowedLine(img, tuple(mid), end, color, 2, cv2.LINE_AA, tipLength=0.4)

    def _draw_blink_flash(
        self, img: np.ndarray, groups: LandmarkGroups
    ) -> None:
        """Flash a ring around each eye on blink."""
        for eye_pts in [groups.left_eye, groups.right_eye]:
            c = eye_pts.mean(axis=0).astype(np.int32)
            r = int(np.linalg.norm(
                eye_pts[:, 0].max() - eye_pts[:, 0].min()
            ) // 2) + 6
            cv2.circle(img, tuple(c), r, (255, 255, 100), 2, cv2.LINE_AA)

    def _draw_multiple_face_warning(
        self, img: np.ndarray, w: int, h: int
    ) -> None:
        """Overlay a red warning banner when multiple faces are detected."""
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, 52), (0, 0, 180), -1)
        cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
        _text(img, "! MULTIPLE FACES DETECTED — FRAME SKIPPED",
              (12, 32), scale=0.65, color=(255, 255, 255), bold=True)

    def _draw_face_absent_overlay(
        self, img: np.ndarray, w: int, h: int
    ) -> None:
        """Dim the frame and show a warning when no face is detected."""
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.45, img, 0.55, 0, img)

    # ── Sidebar builder ───────────────────────────────────────────────────────

    def _build_sidebar(
        self,
        h: int,
        face_detected: bool,
        multiple_faces: bool,
        face_absent_frames: int,
        eye: Optional[EyeContactResult],
        head: Optional[HeadPoseResult],
        smile: Optional[SmileResult],
        emotions: Dict[str, float],
        states: Dict[str, float],
        fps: float,
    ) -> np.ndarray:
        panel = np.full((h, self._sw, 3), cfg.COLOR_BG, dtype=np.uint8)
        sw  = self._sw
        x0  = 14
        bar_w = sw - x0 - 55

        y = 18

        # ── Header ─────────────────────────────────────────────────────────
        _text(panel, "FACIAL ANALYSIS", (x0, y),
              scale=0.55, color=cfg.COLOR_ACCENT, bold=True)
        y += 20
        _divider(panel, x0, y, sw - x0 * 2)
        y += 14

        # ── Face status ─────────────────────────────────────────────────────
        if multiple_faces:
            _text(panel, "! MULTIPLE FACES", (x0, y),
                  scale=0.50, color=cfg.COLOR_WARN, bold=True)
            y += 16
            _text(panel, "  Frame analysis skipped", (x0, y),
                  scale=0.40, color=cfg.COLOR_SUBTEXT)
            y += 20
        elif face_detected:
            _text(panel, "FACE DETECTED  \u2713", (x0, y),
                  scale=0.50, color=cfg.COLOR_ACCENT, bold=True)
            y += 20
        else:
            soft_absent = face_absent_frames >= cfg.FACE_ABSENT_SOFT_FRAMES
            hard_absent = face_absent_frames >= cfg.FACE_ABSENT_HARD_FRAMES
            if hard_absent:
                label = f"NO FACE  ({face_absent_frames} frames)"
            elif soft_absent:
                label = "Face not visible"
            else:
                label = "Detecting face…"
            _text(panel, label, (x0, y), scale=0.48, color=cfg.COLOR_WARN)
            y += 20

        # ── Facial expression estimate ──────────────────────────────────────
        dominant_expr = "—"
        if emotions and face_detected:
            dominant_expr = max(emotions, key=emotions.get).capitalize()
        _text(panel, f"Facial Expr (est.): {dominant_expr}",
              (x0, y), scale=0.44, color=cfg.COLOR_SUBTEXT)
        y += 18

        # ── Eye contact ────────────────────────────────────────────────────
        ec_pct = eye.eye_contact_pct if eye else 0.0
        _text(panel, f"Eye Contact:  {ec_pct:.0f}%", (x0, y),
              scale=0.50, color=cfg.COLOR_TEXT)
        y += 14
        _bar(panel, x0, y, bar_w, 8, ec_pct,
             cfg.COLOR_ACCENT if ec_pct > 50 else cfg.COLOR_WARN)
        y += 18

        # ── Blink rate ─────────────────────────────────────────────────────
        blink_rate  = eye.blink_rate   if eye else 0.0
        blink_total = eye.blink_count  if eye else 0
        _text(panel, f"Blinks:  {blink_rate:.0f}/min  (total {blink_total})",
              (x0, y), scale=0.44, color=cfg.COLOR_TEXT)
        y += 16

        # ── Head pose ───────────────────────────────────────────────────────
        pose_label = head.label if head else "—"
        pose_color = cfg.COLOR_ACCENT if (head and head.is_stable) else cfg.COLOR_WARN
        _text(panel, f"Head Pose:  {pose_label}", (x0, y),
              scale=0.50, color=pose_color)
        y += 18
        if head:
            _text(panel,
                  f"  P:{head.pitch:+.1f}\u00b0 Y:{head.yaw:+.1f}\u00b0 R:{head.roll:+.1f}\u00b0",
                  (x0, y), scale=0.38, color=cfg.COLOR_SUBTEXT)
            y += 14

        # ── Gaze direction ──────────────────────────────────────────────────
        gaze_dir   = eye.gaze_direction if eye else "—"
        gaze_color = cfg.COLOR_ACCENT if gaze_dir == "Center" else (80, 180, 255)
        _text(panel, f"Gaze:  {gaze_dir}", (x0, y), scale=0.50, color=gaze_color)
        y += 18

        # ── Smile score ─────────────────────────────────────────────────────
        smile_score = smile.smile_score if smile else 0.0
        _text(panel, "Smile Score:", (x0, y), scale=0.50, color=cfg.COLOR_TEXT)
        y += 14
        _bar(panel, x0, y, bar_w, 8, smile_score, (220, 200, 60), show_pct=True)
        y += 18

        # ── Facial expression bars ──────────────────────────────────────────
        _divider(panel, x0, y, sw - x0 * 2)
        y += 12
        _text(panel, "FACIAL EXPRESSION (ESTIMATED)", (x0, y),
              scale=0.42, color=cfg.COLOR_ACCENT, bold=True)
        y += 14
        _text(panel, "FACS-inspired geometric estimate", (x0, y),
              scale=0.35, color=cfg.COLOR_SUBTEXT)
        y += 16

        emotion_colors = {
            "happy":    (50, 210, 50),
            "neutral":  (180, 180, 180),
            "sad":      (180, 120, 50),
            "angry":    (40, 60, 220),
            "fear":     (60, 160, 220),
            "surprise": (220, 190, 40),
            "disgust":  (80, 200, 80),
        }
        sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
        for emo_label, emo_val in sorted_emotions[:4]:
            emo_pct = emo_val * 100.0
            color   = emotion_colors.get(emo_label, cfg.COLOR_TEXT)
            _text(panel, f"{emo_label.capitalize()[:7]:<8}", (x0, y),
                  scale=0.38, color=cfg.COLOR_SUBTEXT)
            _bar(panel, x0 + 58, y - 10, bar_w - 58, 12, emo_pct,
                 color, show_pct=True)
            y += 18

        # ── Behavioral pattern estimate ─────────────────────────────────────
        _divider(panel, x0, y, sw - x0 * 2)
        y += 12
        _text(panel, "BEHAVIORAL PATTERN ESTIMATE", (x0, y),
              scale=0.42, color=cfg.COLOR_ACCENT, bold=True)
        y += 14
        _text(panel, "Synthetic-data classifier only", (x0, y),
              scale=0.35, color=cfg.COLOR_SUBTEXT)
        y += 16

        sorted_states = sorted(states.items(), key=lambda x: x[1], reverse=True)
        for state_name, state_pct in sorted_states:
            color = cfg.STATE_COLORS.get(state_name, cfg.COLOR_TEXT)
            # Append "-like" to make the estimate nature explicit
            display_label = f"{state_name}-like"
            _text(panel, f"{display_label:<14}", (x0, y + 1),
                  scale=0.40, color=cfg.COLOR_TEXT)
            _bar(panel, x0 + 100, y - 10, bar_w - 100, 14, state_pct,
                 color, show_pct=True)
            y += 20

        # ── Footer ─────────────────────────────────────────────────────────
        y = h - 26
        _divider(panel, x0, y - 4, sw - x0 * 2)
        fps_color = cfg.COLOR_ACCENT if fps >= 20 else cfg.COLOR_WARN
        _text(panel, f"FPS: {fps:.1f}", (x0, y + 12),
              scale=0.45, color=fps_color)
        _text(panel, "[Q] Quit  [R] Reset", (x0 + 70, y + 12),
              scale=0.38, color=cfg.COLOR_SUBTEXT)

        return panel
