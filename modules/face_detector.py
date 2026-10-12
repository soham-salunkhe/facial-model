"""
modules/face_detector.py
━━━━━━━━━━━━━━━━━━━━━━━━
MediaPipe FaceLandmarker wrapper (MediaPipe 0.10+ / 1.0+ Tasks API).

Responsibilities
----------------
- Initialise MediaPipe Tasks FaceLandmarker using models/face_landmarker.task.
- Auto-download face_landmarker.task model file if missing (~3.6 MB).
- Process BGR frames from OpenCV.
- Return a FaceDetectionResult with an explicit status field:
    FACE_STATUS_NO_FACE   → 0 faces detected
    FACE_STATUS_SINGLE    → exactly 1 face; .face is populated
    FACE_STATUS_MULTIPLE  → >1 faces detected; .face is None (frame is rejected)
- Extract landmark visibility scores where available from the Tasks API.
- 100% compatible with Python 3.13 macOS Apple Silicon.

Phase 2 changes
---------------
- detect() now returns FaceDetectionResult instead of Optional[FaceResult].
- max_num_faces raised to 2 so multiple-face frames can be caught and rejected.
- visibilities array added to FaceResult (float32, 0–1, defaults to 1.0 when
  the MediaPipe model does not expose visibility for a given landmark).
- FaceResult.is_landmark_reliable() checks against cfg.LANDMARK_MIN_VISIBILITY.
"""

from __future__ import annotations

import dataclasses
import os
import urllib.request
from typing import List, Optional

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

import config as cfg

MODEL_TASK_PATH = os.path.join(cfg.MODEL_DIR, "face_landmarker.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)

# ── Face detection status constants ────────────────────────────────────────
FACE_STATUS_NO_FACE  = "no_face"
FACE_STATUS_SINGLE   = "single_face"
FACE_STATUS_MULTIPLE = "multiple_faces"


def _ensure_model_file() -> str:
    """Ensure face_landmarker.task model file exists, downloading if needed."""
    os.makedirs(cfg.MODEL_DIR, exist_ok=True)
    if not os.path.exists(MODEL_TASK_PATH) or os.path.getsize(MODEL_TASK_PATH) < 1000:
        print("[FaceDetector] Downloading face_landmarker.task model file (~3.6 MB)…")
        urllib.request.urlretrieve(MODEL_URL, MODEL_TASK_PATH)
        print("[FaceDetector] Model downloaded successfully.")
    return MODEL_TASK_PATH


@dataclasses.dataclass
class FaceResult:
    """Container for one processed frame's face data (single face only)."""

    # List of raw landmark objects from MediaPipe
    landmarks: object

    # Pixel-space landmark array  shape=(N, 2)  dtype=float32
    pixels: np.ndarray

    # 3-D landmark array (x, y, z normalised)  shape=(N, 3)  dtype=float32
    points3d: np.ndarray

    # Per-landmark visibility scores  shape=(N,)  dtype=float32
    # Range 0.0–1.0. Defaults to 1.0 for each landmark if the MediaPipe
    # FaceLandmarker model does not expose visibility for that point.
    visibilities: np.ndarray

    # Original frame dimensions
    frame_h: int
    frame_w: int

    def pixel(self, idx: int) -> np.ndarray:
        """Return (x, y) pixel coordinates for landmark index `idx`."""
        return self.pixels[idx]

    def point3d(self, idx: int) -> np.ndarray:
        """Return (x, y, z) normalised coordinates for landmark index `idx`."""
        return self.points3d[idx]

    def pixel_list(self, indices: List[int]) -> np.ndarray:
        """Return pixel coords for a list of landmark indices, shape=(N,2)."""
        return self.pixels[indices]

    def is_landmark_reliable(self, idx: int) -> bool:
        """
        Return True if landmark `idx` has visibility >= cfg.LANDMARK_MIN_VISIBILITY.

        Use this to guard geometric operations on specific key landmarks
        (e.g., outer eye corners, mouth corners) when lighting or occlusion
        may have produced an unreliable estimate.
        """
        return float(self.visibilities[idx]) >= cfg.LANDMARK_MIN_VISIBILITY


@dataclasses.dataclass
class FaceDetectionResult:
    """
    Structured result from FaceDetector.detect().

    Attributes
    ----------
    status : str
        One of the FACE_STATUS_* constants:
        - FACE_STATUS_NO_FACE   → no face in frame
        - FACE_STATUS_SINGLE    → exactly one face; see `.face`
        - FACE_STATUS_MULTIPLE  → two or more faces; `.face` is None.
          The pipeline must not process this as a single-candidate frame.
    face : FaceResult | None
        Populated only when status == FACE_STATUS_SINGLE.
    face_count : int
        Raw number of faces detected by MediaPipe.
    """
    status:     str
    face:       Optional[FaceResult] = None
    face_count: int = 0


class FaceDetector:
    """
    Wraps MediaPipe FaceLandmarker for real-time 478-landmark face detection.

    Parameters
    ----------
    max_num_faces : int
        Number of faces to attempt detection on. Set to 2 (default) so that
        multi-face frames can be caught and rejected. Increase only if you
        intentionally need multi-face processing (not recommended for interview use).
    min_detection_confidence : float
        Detection confidence threshold passed to MediaPipe.
    """

    def __init__(
        self,
        max_num_faces: int = 2,            # detect up to 2 to catch multi-face frames
        min_detection_confidence: float = 0.5,
        **kwargs,
    ) -> None:
        model_path = _ensure_model_file()

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=max_num_faces,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_detection_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)

    def detect(self, frame_bgr: np.ndarray) -> FaceDetectionResult:
        """
        Process a BGR frame and return a FaceDetectionResult.

        Parameters
        ----------
        frame_bgr : np.ndarray
            OpenCV BGR frame (H, W, 3).

        Returns
        -------
        FaceDetectionResult
            .status == FACE_STATUS_NO_FACE   → no face found
            .status == FACE_STATUS_SINGLE    → one face; .face is populated
            .status == FACE_STATUS_MULTIPLE  → multiple faces; .face is None
        """
        h, w = frame_bgr.shape[:2]

        # Convert OpenCV BGR → MediaPipe Image (RGB)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        detection_result = self._landmarker.detect(mp_image)
        face_count = len(detection_result.face_landmarks)

        # ── No face ──────────────────────────────────────────────────────────
        if face_count == 0:
            return FaceDetectionResult(
                status=FACE_STATUS_NO_FACE,
                face=None,
                face_count=0,
            )

        # ── Multiple faces — reject frame ────────────────────────────────────
        if face_count > 1:
            return FaceDetectionResult(
                status=FACE_STATUS_MULTIPLE,
                face=None,
                face_count=face_count,
            )

        # ── Single face — extract landmarks ──────────────────────────────────
        face_lms = detection_result.face_landmarks[0]
        n = len(face_lms)

        pixels       = np.zeros((n, 2), dtype=np.float32)
        points3      = np.zeros((n, 3), dtype=np.float32)
        visibilities = np.ones(n, dtype=np.float32)   # default: assume fully visible

        for i, lm in enumerate(face_lms):
            pixels[i]   = [lm.x * w, lm.y * h]
            points3[i]  = [lm.x, lm.y, lm.z]
            # Extract visibility if the Tasks API exposes it for this model.
            # MediaPipe FaceLandmarker may return None for visibility on some builds.
            vis = getattr(lm, "visibility", None)
            if vis is not None:
                visibilities[i] = float(vis)
            # If unavailable, leave at 1.0 (fully reliable assumption).

        return FaceDetectionResult(
            status=FACE_STATUS_SINGLE,
            face=FaceResult(
                landmarks=face_lms,
                pixels=pixels,
                points3d=points3,
                visibilities=visibilities,
                frame_h=h,
                frame_w=w,
            ),
            face_count=1,
        )

    def close(self) -> None:
        """Release MediaPipe resources."""
        if hasattr(self, "_landmarker") and self._landmarker:
            self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
