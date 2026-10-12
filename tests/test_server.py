"""
tests/test_server.py
━━━━━━━━━━━━━━━━━━━━
Unit and integration test suite for the FastAPI & WebSocket Analysis Server.
"""

import base64
import os
import sys
import time
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

# Ensure root directory is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server import app, _pipeline


@pytest.fixture(scope="module")
def client():
    """Create a TestClient with the lifespan event triggered."""
    with TestClient(app) as test_client:
        yield test_client


def make_dummy_jpeg_base64(width: int = 320, height: int = 240) -> str:
    """Create a synthetic test image encoded as base64 JPEG."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    # Draw simple shapes
    cv2.circle(img, (width // 2, height // 2), 40, (120, 180, 240), -1)
    success, buffer = cv2.imencode(".jpg", img)
    assert success
    return base64.b64encode(buffer).decode("utf-8")


def make_dummy_jpeg_bytes(width: int = 320, height: int = 240) -> bytes:
    """Create a synthetic test image encoded as raw JPEG bytes."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.circle(img, (width // 2, height // 2), 40, (120, 180, 240), -1)
    success, buffer = cv2.imencode(".jpg", img)
    assert success
    return buffer.tobytes()


# ── REST Endpoint Tests ───────────────────────────────────────────────────────

def test_health_endpoint(client: TestClient):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["service"] == "facial-analysis-api"
    assert data["pipeline_initialized"] is True
    assert data["active_session"] is not None


def test_session_start_and_status(client: TestClient):
    # Start session with custom ID
    custom_id = "test-session-1234"
    resp = client.post("/session/start", json={"session_id": custom_id})
    assert resp.status_code == 200
    data = resp.json()
    assert data["session_id"] == custom_id
    assert data["is_active"] is True
    assert data["total_frames"] == 0

    # Check status endpoint
    status_resp = client.get("/session/status")
    assert status_resp.status_code == 200
    status_data = status_resp.json()
    assert status_data["session_id"] == custom_id
    assert status_data["total_frames"] == 0


def test_session_reset(client: TestClient):
    # Get current session ID
    initial_status = client.get("/session/status").json()
    initial_id = initial_status["session_id"]

    # Reset session (should retain session_id)
    reset_resp = client.post("/session/reset")
    assert reset_resp.status_code == 200
    reset_data = reset_resp.json()
    assert reset_data["session_id"] == initial_id
    assert reset_data["face_absent_frames"] == 0


def test_analyze_frame_rest(client: TestClient):
    b64_img = make_dummy_jpeg_base64()
    resp = client.post("/analyze/frame", json={"image": b64_img, "timestamp": time.time()})
    assert resp.status_code == 200
    data = resp.json()

    assert "session_id" in data
    assert "frame_index" in data
    assert data["frame_index"] >= 1
    assert "analysis" in data
    assert "processing_ms" in data

    # Verify structured analysis schema (Phase 2 output format)
    analysis = data["analysis"]
    assert "face" in analysis
    assert "eye_contact" in analysis
    assert "head_pose" in analysis
    assert "facial_expression" in analysis
    assert "behavioral_pattern" in analysis

    # Face should be detected as false (dummy image has no face)
    assert analysis["face"]["detected"] is False
    assert analysis["face"]["status"] == "no_face"

    # Behavioral pattern should include synthetic disclaimer note
    assert "_note" in analysis["behavioral_pattern"]
    assert "synthetic" in analysis["behavioral_pattern"]["_note"].lower()


def test_analyze_upload_multipart(client: TestClient):
    jpeg_bytes = make_dummy_jpeg_bytes()
    resp = client.post(
        "/analyze/upload",
        files={"file": ("frame.jpg", jpeg_bytes, "image/jpeg")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "analysis" in data
    assert data["analysis"]["face"]["status"] == "no_face"


def test_invalid_image_payload(client: TestClient):
    resp = client.post("/analyze/frame", json={"image": "invalid-non-base64-content"})
    assert resp.status_code == 400


# ── WebSocket Tests ───────────────────────────────────────────────────────────

def test_websocket_ping_pong(client: TestClient):
    with client.websocket_connect("/ws/analysis") as ws:
        # Initial connect message
        init_msg = ws.receive_json()
        assert init_msg["type"] == "connected"
        assert "session_id" in init_msg

        # Ping
        ws.send_json({"type": "ping"})
        pong_msg = ws.receive_json()
        assert pong_msg["type"] == "pong"


def test_websocket_frame_streaming_base64(client: TestClient):
    b64_img = make_dummy_jpeg_base64()
    with client.websocket_connect("/ws/analysis") as ws:
        init_msg = ws.receive_json()
        assert init_msg["type"] == "connected"

        # Send frame as JSON
        ws.send_json({"type": "frame", "image": b64_img})
        resp = ws.receive_json()

        assert resp["type"] == "analysis_frame"
        assert "data" in resp
        assert "latency_ms" in resp
        assert resp["data"]["face"]["status"] == "no_face"


def test_websocket_binary_frame_streaming(client: TestClient):
    jpeg_bytes = make_dummy_jpeg_bytes()
    with client.websocket_connect("/ws/analysis") as ws:
        init_msg = ws.receive_json()
        assert init_msg["type"] == "connected"

        # Send binary bytes
        ws.send_bytes(jpeg_bytes)
        resp = ws.receive_json()

        assert resp["type"] == "analysis_frame"
        assert "data" in resp
        assert resp["data"]["face"]["status"] == "no_face"


def test_websocket_session_control(client: TestClient):
    with client.websocket_connect("/ws/analysis") as ws:
        init_msg = ws.receive_json()
        assert init_msg["type"] == "connected"

        # Send start command with new ID
        ws.send_json({"type": "start", "session_id": "ws-session-999"})
        started_msg = ws.receive_json()
        assert started_msg["type"] == "session_started"
        assert started_msg["status"]["session_id"] == "ws-session-999"

        # Send reset command
        ws.send_json({"type": "reset"})
        reset_msg = ws.receive_json()
        assert reset_msg["type"] == "session_reset"
        assert reset_msg["status"]["session_id"] == "ws-session-999"


def test_websocket_snapshot_trigger(client: TestClient):
    # Temporarily set snapshot interval to 0.0s to ensure snapshot triggers
    from server import _pipeline
    assert _pipeline is not None
    original_interval = _pipeline.snapshot_interval_sec
    _pipeline.snapshot_interval_sec = 0.0

    try:
        jpeg_bytes = make_dummy_jpeg_bytes()
        with client.websocket_connect("/ws/analysis") as ws:
            ws.receive_json()  # connected

            ws.send_bytes(jpeg_bytes)
            frame_resp = ws.receive_json()
            assert frame_resp["type"] == "analysis_frame"

            # Because snapshot_interval_sec was 0.0, the periodic snapshot message follows
            snapshot_resp = ws.receive_json()
            assert snapshot_resp["type"] == "behavioral_window"
            assert "data" in snapshot_resp
            assert "face_presence_ratio" in snapshot_resp["data"]
    finally:
        _pipeline.snapshot_interval_sec = original_interval
