"""
client_demo.py
━━━━━━━━━━━━━━
Demo client demonstrating WebSocket streaming to the AI Facial Analysis Service.

Can stream:
  1. Synthetic test frames (no webcam needed: python client_demo.py --synthetic)
  2. Local webcam feed (python client_demo.py --cam 0)

Usage:
  # First run the server in a separate terminal:
  python server.py --port 8000

  # Then run the client:
  python client_demo.py --synthetic --frames 60
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import cv2
import numpy as np
import websockets


def generate_synthetic_frame(frame_idx: int, width: int = 640, height: int = 480) -> bytes:
    """Generate a synthetic test image with motion."""
    frame = np.full((height, width, 3), (24, 24, 30), dtype=np.uint8)
    # Draw animated circle
    x = int(width / 2 + 100 * np.sin(frame_idx * 0.1))
    y = int(height / 2 + 50 * np.cos(frame_idx * 0.1))
    cv2.circle(frame, (x, y), 50, (80, 200, 120), -1)
    cv2.putText(
        frame,
        f"Frame {frame_idx}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
    )
    success, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buffer.tobytes()


async def run_client(
    uri: str = "ws://localhost:8000/ws/analysis",
    synthetic: bool = True,
    cam_index: int = 0,
    max_frames: int = 150,
) -> None:
    print(f"[client] Connecting to {uri}…")
    async with websockets.connect(uri) as ws:
        # Await connection acknowledgment
        init_raw = await ws.recv()
        init_data = json.loads(init_raw)
        print(f"[client] Connected! Active session: {init_data.get('session_id')}")

        cap = None
        if not synthetic:
            cap = cv2.VideoCapture(cam_index)
            if not cap.isOpened():
                print(f"[ERROR] Could not open webcam {cam_index}, falling back to synthetic.")
                synthetic = True

        frames_sent = 0
        t_start = time.perf_counter()

        try:
            while frames_sent < max_frames:
                frames_sent += 1

                if synthetic or cap is None:
                    jpeg_bytes = generate_synthetic_frame(frames_sent)
                else:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    jpeg_bytes = buffer.tobytes()

                # Send binary JPEG frame over WebSocket
                await ws.send(jpeg_bytes)

                # Receive analysis response
                resp_raw = await ws.recv()
                resp = json.loads(resp_raw)

                if resp.get("type") == "analysis_frame":
                    latency = resp.get("latency_ms", 0.0)
                    face_status = resp.get("data", {}).get("face", {}).get("status")
                    pattern = resp.get("data", {}).get("behavioral_pattern", {}).get("dominant_pattern")
                    if frames_sent % 30 == 0 or frames_sent <= 3:
                        print(
                            f"[client] Frame #{frames_sent:03d} | "
                            f"Latency: {latency:.1f}ms | "
                            f"Face: {face_status} | "
                            f"Pattern: {pattern}"
                        )

                # Check if a behavioral window message was also emitted
                # (handled non-blocking or sequential check)
                try:
                    extra_raw = await asyncio.wait_for(ws.recv(), timeout=0.005)
                    extra = json.loads(extra_raw)
                    if extra.get("type") == "behavioral_window":
                        snap = extra.get("data", {})
                        print(
                            f"\n>>> [client] BEHAVIORAL WINDOW SNAPSHOT (duration={snap.get('window_duration')}s):"
                        )
                        print(f"    Presence: {snap.get('face_presence_ratio') * 100:.1f}%")
                        print(f"    Eye Contact Mean: {snap.get('eye_contact_mean')}%")
                        print(f"    Posture Stability: {snap.get('posture_stability')}\n")
                except asyncio.TimeoutError:
                    pass

                # Cap client send rate to ~30 FPS
                await asyncio.sleep(1.0 / 30.0)

        finally:
            if cap is not None:
                cap.release()

        elapsed = time.perf_counter() - t_start
        fps = frames_sent / elapsed if elapsed > 0 else 0
        print(f"[client] Completed streaming {frames_sent} frames in {elapsed:.2f}s ({fps:.1f} FPS).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WebSocket Streaming Client Demo")
    parser.add_argument("--uri", default="ws://localhost:8000/ws/analysis", help="WebSocket URI")
    parser.add_argument("--synthetic", action="store_true", default=True, help="Use synthetic frames")
    parser.add_argument("--cam", type=int, default=0, help="Webcam index if not synthetic")
    parser.add_argument("--frames", type=int, default=60, help="Number of frames to stream")
    args = parser.parse_args()

    asyncio.run(run_client(
        uri=args.uri,
        synthetic=args.synthetic,
        cam_index=args.cam,
        max_frames=args.frames,
    ))
