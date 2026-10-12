# AI-Powered Facial Analysis Module
### Mock Interview System — Real-Time Face Analysis Service

> Analyzes a candidate's face via webcam or streaming frames in real time and predicts observable behavioral patterns using a fusion of 478 MediaPipe facial landmarks, FACS-inspired Action Unit geometry, iris gaze tracking, EMA-smoothed head pose estimation, and a rolling behavioral window.

---

## Architecture Overview

```
Webcam Frame / WebSocket Binary / REST Upload
                       │
                       ▼
                 FaceDetector (MediaPipe Tasks 478 Landmarks)
                       │
         ┌─────────────┴─────────────┐
         ▼                           ▼
[No / Multiple Faces]         [Single Face]
         │                           │
         │             LandmarkExtractor
         │                     │
         │       ┌─────────────┼─────────────┬─────────────┐
         │       ▼             ▼             ▼             ▼
         │  EyeContact     HeadPose        Smile       EmotionDetector
         │  (EAR + Gaze) (solvePnP EMA)  (Mouth MAR)  (FACS Action Units)
         │       │             │             │             │
         │       └─────────────┼─────────────┴─────────────┘
         │                     ▼
         │              FeatureAggregator (22-D vector)
         │                     ▼
         │             InterviewClassifier (XGBoost)
         │                     │
         └─────────────┬───────┘
                       ▼
          BehavioralWindowAggregator (7-Second Rolling Window)
                       │
         ┌─────────────┴─────────────┐
         ▼                           ▼
build_analysis_frame()       LiveDashboard (OpenCV HUD)
         │
         ▼
FastAPI / WebSocket Server (/ws/analysis, 25–30 FPS)
```

---

## Features & Modules

| Module | Technology | What it does |
|---|---|---|
| **Face Detection** | MediaPipe Face Landmarker | Detects up to 2 faces; enforces single-candidate mode; extracts 478 landmarks with reliability tracking |
| **Landmark Extraction** | NumPy indexing | Groups landmarks into semantic clusters (eyes, eyebrows, mouth, jawline) |
| **Eye Contact** | EAR formula + iris centroid | Blink detection, gaze direction, eye contact percentage, elapsed session tracking |
| **Head Pose** | OpenCV solvePnP + EMA | Pitch / Yaw / Roll in degrees with $\alpha = 0.35$ exponential smoothing |
| **Smile Detection** | Mouth corner geometry | 0–100 smile score with baseline drift clamping (`[-0.05, 0.15]`) |
| **Facial Expression** | FACS-inspired Action Units | AU1, AU2, AU4, AU6, AU12, AU15, AU25 geometric expression estimates |
| **Feature Fusion** | FeatureAggregator | 22-D feature vector per frame with motion spike protection on face reappearance |
| **Behavioral Pattern** | XGBoost classifier | Evaluates behavioral archetype resemblance (`confident-like`, `nervous-like`, etc.) |
| **Behavioral Window** | Rolling deque aggregator | 7-second rolling window computing eye contact mean/std, posture stability, presence ratio |
| **FastAPI / WebSocket** | FastAPI + Uvicorn + WebSockets | High-performance streaming server emitting per-frame analysis and window snapshots |
| **Live Dashboard** | OpenCV HUD | Real-time desktop GUI with metrics, overlays, and multiple-face warning banners |

---

## Quick Start

### 1. Install Dependencies

```bash
cd "facial model"
pip install -r requirements.txt
```

### 2. Option A: Run Local OpenCV Desktop GUI

```bash
python main.py
```

Controls:
- `Q` — Quit
- `R` — Reset session (blink counters, baseline, temporal motion state, rolling window)
- `S` — Save screenshot to `screenshots/`

Command-Line Options:
```bash
python main.py --cam 0        # Webcam device index
python main.py --width 1280   # Capture width
python main.py --height 720   # Capture height
python main.py --retrain      # Force retrain classifier
python main.py --no-train     # Skip auto-training
```

---

### 3. Option B: Run FastAPI & WebSocket Server (Phase 3)

Start the server:
```bash
python server.py --host 0.0.0.0 --port 8000
```
Or with Uvicorn CLI:
```bash
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

#### API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check, active session UUID, and status |
| `POST` | `/session/start` | Starts new session, resets baselines, returns session metadata |
| `POST` | `/session/reset` | Resets metrics and window while retaining session ID |
| `GET` | `/session/status` | Returns session progress, duration, and rolling behavioral snapshot |
| `POST` | `/analyze/frame` | Synchronous REST endpoint accepting base64 image JSON |
| `POST` | `/analyze/upload` | Synchronous REST endpoint accepting multipart image upload |
| `WS` | `/ws/analysis` | Bidirectional WebSocket streaming endpoint (~25–30 FPS) |

#### WebSocket Protocol (`/ws/analysis`)

**Client to Server:**
- **Binary JPEG/PNG Bytes**: Directly stream raw image bytes for low bandwidth and latency.
- **JSON Frame**: `{"type": "frame", "image": "<base64>", "timestamp": 1728409000.12}`
- **JSON Commands**:
  - `{"type": "ping"}`
  - `{"type": "start", "session_id": "optional-id"}`
  - `{"type": "reset"}`
  - `{"type": "status"}`

**Server to Client:**
- **Per-frame analysis** (every frame, ~25–30 FPS):
  ```json
  {
    "type": "analysis_frame",
    "session_id": "...",
    "frame_index": 42,
    "latency_ms": 11.4,
    "data": {
      "timestamp": 1728409000.12,
      "face": { "detected": true, "status": "single_face", "face_count": 1 },
      "eye_contact": { "percentage": 82.5, "gaze": "center", "blink_rate_per_min": 18.0, ... },
      "head_pose": { "pitch": -1.2, "yaw": 2.4, "roll": 0.5, "is_stable": true, ... },
      "facial_expression": { "expression_estimate": "neutral", "_note": "..." },
      "behavioral_pattern": { "dominant_pattern": "confident-like", "_note": "..." }
    }
  }
  ```
- **Behavioral window snapshot** (emitted every ~5 seconds):
  ```json
  {
    "type": "behavioral_window",
    "session_id": "...",
    "timestamp": 1728409005.12,
    "data": {
      "window_duration": 5.0,
      "eye_contact_mean": 81.2,
      "eye_contact_std": 4.1,
      "head_yaw_mean": 1.8,
      "smile_mean": 12.0,
      "posture_stability": 0.95,
      "face_presence_ratio": 1.0
    }
  }
  ```

---

### 4. Run Streaming Client Demo

In a separate terminal:
```bash
# Test with synthetic animated frames (no webcam needed)
python client_demo.py --synthetic --frames 60

# Or stream local webcam over WebSocket
python client_demo.py --cam 0 --frames 150
```

---

### 5. Running the Automated Test Suite

```bash
pytest tests/test_server.py -v
```
All 11 unit and integration tests validate REST endpoints, multipart decoding, WebSocket ping/pong, JSON frame streaming, binary frame streaming, and behavioral snapshot triggering.

---

## Project Structure

```
facial model/
├── main.py                          # OpenCV desktop HUD entry point
├── server.py                        # FastAPI & WebSocket server
├── client_demo.py                   # WebSocket streaming client demo
├── config.py                        # Central parameters, thresholds & server config
├── requirements.txt                 # Dependencies
│
├── modules/
│   ├── __init__.py                  # Package exports
│   ├── pipeline.py                  # Canonical FacialAnalysisPipeline engine
│   ├── face_detector.py             # MediaPipe FaceLandmarker with multi-face support
│   ├── landmark_extractor.py        # Semantic landmark groups
│   ├── eye_contact.py               # EAR, blink rate (session-start fix), gaze
│   ├── head_pose.py                 # solvePnP + EMA pitch/yaw/roll smoothing
│   ├── emotion_detector.py          # FACS Action Units geometric expression
│   ├── smile_detector.py            # Baseline-clamped smile score
│   ├── feature_aggregator.py        # 22-D vector fusion + temporal reset
│   ├── interview_classifier.py      # XGBoost behavioral archetype estimator
│   ├── behavioral_window.py         # 7-second rolling window aggregator
│   └── analysis_output.py           # Structured JSON output formatter
│
├── dashboard/
│   └── live_dashboard.py            # OpenCV HUD renderer with relabelled terminology
│
├── tests/
│   └── test_server.py               # Pytest suite for FastAPI and WebSockets
│
├── training/
│   ├── synthetic_data.py            # Synthetic rule generator
│   └── train_classifier.py          # Standalone training script
│
└── models/                          # Auto-created on first run
    ├── interview_classifier.pkl
    └── scaler.pkl
```

---

## Ethical & Scientific Transparency Notice

1. **Observational Geometry**: Eye contact %, gaze direction, head pose angles, and smile metrics are computed geometrically from facial landmark coordinates.
2. **Action Unit Expression Estimates**: Expression scores are FACS-inspired geometric approximations, not clinically validated emotion detectors.
3. **Behavioral Archetypes**: The XGBoost classifier is trained on synthetic data and outputs archetype resemblance scores (`confident-like`, `nervous-like`). They represent pattern resemblance and do not measure candidate psychology, competence, or veracity.
# facial-model
