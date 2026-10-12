"""
config.py — Central configuration for the Facial Analysis Module.
All thresholds, landmark indices, model paths, and tunable constants live here.
"""

# ─────────────────────────────────────────────
# MediaPipe Landmark Index Groups
# ─────────────────────────────────────────────

# Left eye contour (16 points)
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173]
# Right eye contour (16 points)
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

# EAR landmark sub-indices within the eye arrays above (p1,p2,p3,p4,p5,p6)
LEFT_EAR_IDX  = [33, 160, 158, 133, 153, 144]   # horizontal corners + vertical pairs
RIGHT_EAR_IDX = [362, 385, 387, 263, 373, 380]

# Iris centers (refine_landmarks=True required)
LEFT_IRIS  = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]

# Eyebrow landmarks
LEFT_EYEBROW  = [70, 63, 105, 66, 107]
RIGHT_EYEBROW = [336, 296, 334, 293, 300]

# Mouth outer contour
MOUTH_OUTER = [78, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191, 95]
# Key mouth landmarks for MAR
MOUTH_MAR_IDX = [13, 14, 78, 308, 82, 87]  # top, bottom, left, right, upper-mid, lower-mid

# Nose tip
NOSE_TIP = 4

# Jawline keypoints
JAWLINE = [152, 58, 132, 361, 288, 10, 234, 454]

# Landmark indices used for head pose solvePnP
HEAD_POSE_POINTS = {
    "nose_tip":         4,
    "chin":             152,
    "left_eye_outer":   33,
    "right_eye_outer":  263,
    "mouth_left":       61,
    "mouth_right":      291,
}

# ─────────────────────────────────────────────
# Eye Contact & Blink Thresholds
# ─────────────────────────────────────────────
EAR_BLINK_THRESHOLD     = 0.22    # EAR below this → eye closed
EAR_CONSEC_FRAMES       = 3       # consecutive frames below threshold → blink
GAZE_CENTER_THRESHOLD   = 0.20    # iris offset fraction for "center" gaze
EYE_CONTACT_WINDOW_SEC  = 5.0     # rolling window for eye contact %
BLINK_RATE_WINDOW_SEC   = 60.0    # rolling window for blinks-per-minute

# ─────────────────────────────────────────────
# Head Pose Thresholds (degrees)
# ─────────────────────────────────────────────
PITCH_UP_THRESHOLD    = -15.0   # looking up
PITCH_DOWN_THRESHOLD  =  15.0   # looking down
YAW_LEFT_THRESHOLD    = -20.0   # looking left
YAW_RIGHT_THRESHOLD   =  20.0   # looking right
ROLL_TILT_THRESHOLD   =  15.0   # head tilt

# ─────────────────────────────────────────────
# Smile & Mouth Thresholds
# ─────────────────────────────────────────────
SMILE_CORNER_THRESHOLD = 0.02   # normalized vertical rise of mouth corners
MAR_OPEN_THRESHOLD     = 0.35   # mouth aspect ratio for open mouth

# ─────────────────────────────────────────────
# Emotion Detection
# ─────────────────────────────────────────────
EMOTION_THREAD_SKIP  = 5        # process emotion every N frames
EMOTION_ALPHA        = 0.4      # EMA smoothing factor (0=slow, 1=fast)
EMOTION_LABELS       = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# ─────────────────────────────────────────────
# Interview State Classifier
# ─────────────────────────────────────────────
STATE_LABELS           = ["Confident", "Nervous", "Confused", "Stressed", "Happy"]
STATE_SMOOTH_FRAMES    = 10      # rolling average window for output probs
FEATURE_DIM            = 22     # dimensionality of feature vector

# ─────────────────────────────────────────────
# Model Paths
# ─────────────────────────────────────────────
import os
_BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR             = os.path.join(_BASE, "models")
CLASSIFIER_PATH       = os.path.join(MODEL_DIR, "interview_classifier.pkl")
SCALER_PATH           = os.path.join(MODEL_DIR, "scaler.pkl")

# ─────────────────────────────────────────────
# Dashboard & Rendering
# ─────────────────────────────────────────────
WINDOW_TITLE    = "Facial Analysis — Mock Interview System"
SIDEBAR_WIDTH   = 380           # pixels
CAM_WIDTH       = 1280
CAM_HEIGHT      = 720
TARGET_FPS      = 30

# Color palette (BGR)
COLOR_BG           = (18, 18, 24)
COLOR_ACCENT       = (80, 200, 120)      # green
COLOR_WARN         = (60, 120, 255)      # orange-red (BGR)
COLOR_TEXT         = (235, 235, 235)
COLOR_SUBTEXT      = (140, 140, 160)
COLOR_BAR_BG       = (45, 45, 55)
COLOR_CONFIDENT    = (100, 220, 100)
COLOR_NERVOUS      = (60, 120, 255)
COLOR_CONFUSED     = (60, 200, 255)
COLOR_STRESSED     = (40, 80, 220)
COLOR_HAPPY        = (220, 200, 60)
COLOR_MESH         = (0, 200, 100)
COLOR_IRIS         = (255, 200, 0)

STATE_COLORS = {
    "Confident": COLOR_CONFIDENT,
    "Nervous":   COLOR_NERVOUS,
    "Confused":  COLOR_CONFUSED,
    "Stressed":  COLOR_STRESSED,
    "Happy":     COLOR_HAPPY,
}

# ─────────────────────────────────────────────
# Head Pose Temporal Smoothing
# ─────────────────────────────────────────────
HEAD_POSE_SMOOTH_ALPHA  = 0.35   # EMA alpha for pitch/yaw/roll (0=slow, 1=instant)

# ─────────────────────────────────────────────
# Face Absence Handling
# ─────────────────────────────────────────────
FACE_ABSENT_SOFT_FRAMES = 5     # frames of absence before "soft hold" warning
FACE_ABSENT_HARD_FRAMES = 30    # frames before "hard absent" — reset stale states

# ─────────────────────────────────────────────
# Smile Baseline Drift Protection
# ─────────────────────────────────────────────
SMILE_BASELINE_MIN = -0.05   # floor for neutral corner_raise baseline
SMILE_BASELINE_MAX =  0.15   # ceiling — prevents drift during sustained smiling

# ─────────────────────────────────────────────
# Behavioral Window Aggregation
# ─────────────────────────────────────────────
BEHAVIORAL_WINDOW_SEC = 7.0  # rolling window duration in seconds

# ─────────────────────────────────────────────
# Landmark Quality
# ─────────────────────────────────────────────
LANDMARK_MIN_VISIBILITY = 0.1  # landmarks below this are flagged as low-confidence

# ─────────────────────────────────────────────
# Server & Streaming (Phase 3)
# ─────────────────────────────────────────────
SERVER_HOST             = "0.0.0.0"
SERVER_PORT             = 8000
WS_SNAPSHOT_INTERVAL_SEC = 5.0  # seconds between emitting behavioral window snapshots

