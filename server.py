"""
server.py
━━━━━━━━━
FastAPI & WebSocket server for real-time AI Facial Analysis.

Provides:
  - WebSocket streaming endpoint (/ws/analysis) emitting per-frame analysis
    JSON at 25–30 FPS and rolling BehavioralWindowSnapshot every 5 seconds.
  - REST endpoints for session lifecycle:
      POST /session/start
      POST /session/reset
      GET  /session/status
      POST /analyze/frame (single-frame REST inspection)
      GET  /health
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import sys
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

import cv2
import numpy as np
from fastapi import (
    FastAPI,
    File,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as cfg
from modules import (
    FacialAnalysisPipeline,
    PipelineFrameResult,
    BehavioralWindowSnapshot,
)

# ── Global Pipeline Instance & Lock ──────────────────────────────────────────
_pipeline: Optional[FacialAnalysisPipeline] = None
_pipeline_lock = asyncio.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager: initialize CV pipeline on startup and clean up on shutdown."""
    global _pipeline
    print("[server] Initialising Facial Analysis Pipeline…")
    _pipeline = FacialAnalysisPipeline(
        auto_train_classifier=True,
        snapshot_interval_sec=cfg.WS_SNAPSHOT_INTERVAL_SEC,
    )
    print(f"[server] Pipeline ready. Active session: {_pipeline.session_id}")
    yield
    print("[server] Shutting down — releasing pipeline resources…")
    if _pipeline is not None:
        _pipeline.close()
        _pipeline = None
    print("[server] Shutdown complete.")


# ── FastAPI App Setup ────────────────────────────────────────────────────────
app = FastAPI(
    title="AI Facial Analysis Service",
    description="Real-time computer-vision facial behavior and expression analysis API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic Request / Response Schemas ──────────────────────────────────────
class SessionStartRequest(BaseModel):
    session_id: Optional[str] = Field(None, description="Optional custom session UUID")
    reset_baselines: bool = Field(True, description="Whether to reset smiling and temporal baselines")


class SessionResponse(BaseModel):
    session_id: str
    is_active: bool
    started_at: float
    elapsed_seconds: float
    total_frames: int
    face_absent_frames: int
    is_face_present: bool
    behavioral_snapshot: Optional[Dict[str, Any]] = None


class FrameAnalyzeRequest(BaseModel):
    image: str = Field(..., description="Base64 encoded JPEG or PNG image (with or without data URI header)")
    timestamp: Optional[float] = Field(None, description="Client capture timestamp in seconds")


class FrameAnalyzeResponse(BaseModel):
    session_id: str
    frame_index: int
    analysis: Dict[str, Any]
    behavioral_snapshot: Optional[Dict[str, Any]] = None
    processing_ms: float


# ── Image Decoding Helpers ───────────────────────────────────────────────────
def decode_base64_image(base64_str: str) -> np.ndarray:
    """Decode a base64-encoded image string to a BGR numpy array."""
    try:
        # Strip data URL prefix if present (e.g. data:image/jpeg;base64,...)
        if "," in base64_str:
            base64_str = base64_str.split(",", 1)[1]

        img_bytes = base64.b64decode(base64_str)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("cv2.imdecode returned None")
        return frame
    except Exception as e:
        raise ValueError(f"Failed to decode base64 image: {e}") from e


def decode_bytes_image(img_bytes: bytes) -> np.ndarray:
    """Decode raw image bytes to a BGR numpy array."""
    try:
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("cv2.imdecode returned None")
        return frame
    except Exception as e:
        raise ValueError(f"Failed to decode binary image bytes: {e}") from e


# ── REST Endpoints ───────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
async def health_check() -> Dict[str, Any]:
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "facial-analysis-api",
        "version": "1.0.0",
        "pipeline_initialized": _pipeline is not None,
        "active_session": _pipeline.session_id if _pipeline else None,
    }


@app.post("/session/start", response_model=SessionResponse, tags=["Session"])
async def session_start(req: Optional[SessionStartRequest] = None) -> SessionResponse:
    """
    Start or re-initialize an interview session.
    Resets rolling counters, gaze tracking, smile baseline, and window aggregator.
    """
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    async with _pipeline_lock:
        status_info = _pipeline.reset_session(new_session_id=True)
        if req and req.session_id:
            _pipeline.session_id = req.session_id
            status_info["session_id"] = req.session_id

    return SessionResponse(**status_info)


@app.post("/session/reset", response_model=SessionResponse, tags=["Session"])
async def session_reset() -> SessionResponse:
    """
    Reset metrics, temporal states, and rolling window while maintaining current session ID.
    """
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    async with _pipeline_lock:
        status_info = _pipeline.reset_session(new_session_id=False)

    return SessionResponse(**status_info)


@app.get("/session/status", response_model=SessionResponse, tags=["Session"])
async def session_status() -> SessionResponse:
    """
    Get current interview session progress, frame counts, and rolling window summary.
    """
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    async with _pipeline_lock:
        status_info = _pipeline.get_session_status()

    return SessionResponse(**status_info)


@app.post("/analyze/frame", response_model=FrameAnalyzeResponse, tags=["Analysis"])
async def analyze_frame_json(req: FrameAnalyzeRequest) -> FrameAnalyzeResponse:
    """
    Analyze a single base64-encoded frame synchronously.
    Useful for testing, periodic snapshots, or REST-based integration.
    """
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        frame = decode_base64_image(req.image)
    except ValueError as err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(err))

    start_t = time.perf_counter()
    async with _pipeline_lock:
        # Offload synchronous CV computation to worker thread to keep event loop unblocked
        result: PipelineFrameResult = await asyncio.to_thread(
            _pipeline.process_frame, frame, req.timestamp
        )
    elapsed_ms = (time.perf_counter() - start_t) * 1000.0

    snapshot_dict = result.snapshot._asdict() if result.snapshot else None

    return FrameAnalyzeResponse(
        session_id=_pipeline.session_id,
        frame_index=_pipeline.total_frames,
        analysis=result.analysis,
        behavioral_snapshot=snapshot_dict,
        processing_ms=round(elapsed_ms, 2),
    )


@app.post("/analyze/upload", response_model=FrameAnalyzeResponse, tags=["Analysis"])
async def analyze_frame_upload(file: UploadFile = File(...)) -> FrameAnalyzeResponse:
    """
    Analyze a single uploaded image file (multipart/form-data).
    """
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    file_bytes = await file.read()
    try:
        frame = decode_bytes_image(file_bytes)
    except ValueError as err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(err))

    start_t = time.perf_counter()
    async with _pipeline_lock:
        result: PipelineFrameResult = await asyncio.to_thread(
            _pipeline.process_frame, frame
        )
    elapsed_ms = (time.perf_counter() - start_t) * 1000.0

    snapshot_dict = result.snapshot._asdict() if result.snapshot else None

    return FrameAnalyzeResponse(
        session_id=_pipeline.session_id,
        frame_index=_pipeline.total_frames,
        analysis=result.analysis,
        behavioral_snapshot=snapshot_dict,
        processing_ms=round(elapsed_ms, 2),
    )


# ── WebSocket Streaming Endpoint ─────────────────────────────────────────────
@app.websocket("/ws/analysis")
async def websocket_analysis_endpoint(websocket: WebSocket) -> None:
    """
    Real-time bidirectional WebSocket streaming endpoint.

    Supported Client Messages:
      1. Binary Frame: Raw JPEG / PNG bytes.
      2. Text JSON Frame: {"type": "frame", "image": "<base64>", "timestamp": <float>}
      3. Control JSON:
         - {"type": "reset"}  → resets session baselines
         - {"type": "start", "session_id": "..."} → starts new session
         - {"type": "status"} → requests session status
         - {"type": "ping"}   → responds with pong

    Server Emissions:
      1. Per-frame analysis (type: "analysis_frame"):
         {
           "type": "analysis_frame",
           "session_id": "...",
           "frame_index": 120,
           "latency_ms": 11.2,
           "data": { ...build_analysis_frame dict... }
         }
      2. Periodic behavioral window (type: "behavioral_window") emitted every ~5s:
         {
           "type": "behavioral_window",
           "session_id": "...",
           "timestamp": 1728409000.12,
           "data": { ...BehavioralWindowSnapshot dict... }
         }
    """
    await websocket.accept()
    if _pipeline is None:
        await websocket.send_json({"type": "error", "message": "Pipeline not initialized"})
        await websocket.close()
        return

    # Send initial connection acknowledgment
    await websocket.send_json({
        "type": "connected",
        "session_id": _pipeline.session_id,
        "snapshot_interval_sec": _pipeline.snapshot_interval_sec,
    })

    try:
        while True:
            # Receive either text (JSON / base64) or binary data
            message = await websocket.receive()
            frame: Optional[np.ndarray] = None
            custom_timestamp: Optional[float] = None

            if "text" in message and message["text"] is not None:
                text_data = message["text"].strip()

                # Attempt JSON command parse
                if text_data.startswith("{") and text_data.endswith("}"):
                    try:
                        payload = json.loads(text_data)
                    except json.JSONDecodeError:
                        await websocket.send_json({"type": "error", "message": "Invalid JSON payload"})
                        continue

                    msg_type = payload.get("type", "frame")

                    if msg_type == "ping":
                        await websocket.send_json({"type": "pong", "timestamp": time.time()})
                        continue

                    elif msg_type == "reset":
                        async with _pipeline_lock:
                            status_dict = _pipeline.reset_session(new_session_id=False)
                        await websocket.send_json({"type": "session_reset", "status": status_dict})
                        continue

                    elif msg_type == "start":
                        sid = payload.get("session_id")
                        async with _pipeline_lock:
                            status_dict = _pipeline.reset_session(new_session_id=True)
                            if sid:
                                _pipeline.session_id = sid
                                status_dict["session_id"] = sid
                        await websocket.send_json({"type": "session_started", "status": status_dict})
                        continue

                    elif msg_type == "status":
                        async with _pipeline_lock:
                            status_dict = _pipeline.get_session_status()
                        await websocket.send_json({"type": "session_status", "status": status_dict})
                        continue

                    elif msg_type == "frame":
                        img_str = payload.get("image")
                        custom_timestamp = payload.get("timestamp")
                        if not img_str:
                            await websocket.send_json({"type": "error", "message": "Missing 'image' in frame message"})
                            continue
                        try:
                            frame = decode_base64_image(img_str)
                        except ValueError as err:
                            await websocket.send_json({"type": "error", "message": str(err)})
                            continue
                    else:
                        await websocket.send_json({"type": "error", "message": f"Unknown message type '{msg_type}'"})
                        continue

                else:
                    # Treat raw text as base64 image
                    try:
                        frame = decode_base64_image(text_data)
                    except ValueError as err:
                        await websocket.send_json({"type": "error", "message": str(err)})
                        continue

            elif "bytes" in message and message["bytes"] is not None:
                # Binary frame data
                try:
                    frame = decode_bytes_image(message["bytes"])
                except ValueError as err:
                    await websocket.send_json({"type": "error", "message": str(err)})
                    continue

            else:
                # Connection close or ping
                continue

            if frame is None:
                continue

            # Process frame through pipeline
            t_start = time.perf_counter()
            async with _pipeline_lock:
                result: PipelineFrameResult = await asyncio.to_thread(
                    _pipeline.process_frame, frame, custom_timestamp
                )
            latency_ms = (time.perf_counter() - t_start) * 1000.0

            # 1. Send per-frame analysis JSON
            await websocket.send_json({
                "type": "analysis_frame",
                "session_id": _pipeline.session_id,
                "frame_index": _pipeline.total_frames,
                "latency_ms": round(latency_ms, 2),
                "data": result.analysis,
            })

            # 2. Check if periodic 5-second behavioral window snapshot is due
            if result.should_emit_snapshot and result.snapshot is not None:
                await websocket.send_json({
                    "type": "behavioral_window",
                    "session_id": _pipeline.session_id,
                    "timestamp": round(time.time(), 3),
                    "data": result.snapshot._asdict(),
                })

    except WebSocketDisconnect:
        # Normal client disconnection
        pass
    except Exception as exc:
        try:
            await websocket.send_json({"type": "error", "message": f"Server error: {exc}"})
        except Exception:
            pass


# ── Standalone CLI Runner ────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="AI Facial Analysis Server")
    parser.add_argument("--host", default=cfg.SERVER_HOST, help=f"Host (default {cfg.SERVER_HOST})")
    parser.add_argument("--port", type=int, default=cfg.SERVER_PORT, help=f"Port (default {cfg.SERVER_PORT})")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")

    args = parser.parse_args()
    print(f"Starting server on http://{args.host}:{args.port}")
    uvicorn.run("server:app", host=args.host, port=args.port, reload=args.reload)
