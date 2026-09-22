import asyncio
import os
import time
import cv2
import numpy as np
import torch
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, status
from pydantic import BaseModel, Field
from supabase import Client, create_client
from ultralytics import YOLO

load_dotenv()

SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

app = FastAPI(title="Suraksha Core Engine")

# Load model and move to CUDA if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO("yolo11n-pose.pt")
# Optional export for speedup: model.export(format="engine") or format="onnx"


class BoundaryBox(BaseModel):
    x_min: float = Field(..., ge=0.0, le=1.0)
    y_min: float = Field(..., ge=0.0, le=1.0)
    width: float = Field(..., ge=0.0, le=1.0)
    height: float = Field(..., ge=0.0, le=1.0)


class ThreatBox(BaseModel):
    id: str
    threat_score: float = Field(..., ge=0.0, le=1.0)
    distance_m: float
    bbox: BoundaryBox


class DataPayload(BaseModel):
    timestamp_ms: int
    targets: list[ThreatBox] = []


def run_inference(frame: np.ndarray):
    with torch.inference_mode():
        return model.track(
            source=frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=[0],
            imgsz=320, 
            device=device,
            verbose=False,
        )


@app.websocket("/prod/main")
async def main_suraksha(websocket: WebSocket, token: str = Query(None)):
    # --- Authentication Logic (Unchanged) ---
    if not token:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Missing authorization token")
        return

    try:
        user_response = await asyncio.to_thread(supabase.auth.get_user, token)
        if not user_response or not user_response.user:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Invalid token")
            return
        user = user_response.user
    except Exception as e:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason=f"Authentication failed: {str(e)}")
        return

    await websocket.accept()

    # Create a single-slot buffer to hold only the absolute latest frame
    latest_frame_bytes = None
    new_frame_event = asyncio.Event()

    async def frame_receiver():
        nonlocal latest_frame_bytes
        try:
            while True:
                jpeg_bytes = await websocket.receive_bytes()
                latest_frame_bytes = jpeg_bytes  
                new_frame_event.set()
        except Exception:
            pass

    receiver_task = asyncio.create_task(frame_receiver())

    try:
        while True:
            await new_frame_event.wait()
            new_frame_event.clear()

            if latest_frame_bytes is None:
                continue

            current_bytes = latest_frame_bytes
            latest_frame_bytes = None

            np_arr = np.frombuffer(current_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if frame is None:
                continue

            results = await asyncio.to_thread(run_inference, frame)
            targets: list[ThreatBox] = []

            if results and len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
                boxes_obj = results[0].boxes
                boxes = boxes_obj.xywhn.cpu().numpy()

                track_ids = (
                    boxes_obj.id.cpu().numpy().astype(int)
                    if boxes_obj.id is not None
                    else list(range(len(boxes)))
                )

                scores = (
                    boxes_obj.conf.cpu().numpy()
                    if boxes_obj.conf is not None
                    else [0.0] * len(boxes)
                )

                for idx, track_id in enumerate(track_ids):
                    cx, cy, w, h = boxes[idx]

                    x_min = max(0.0, min(1.0, float(cx - (w / 2.0))))
                    y_min = max(0.0, min(1.0, float(cy - (h / 2.0))))
                    norm_w = max(0.0, min(1.0, float(w)))
                    norm_h = max(0.0, min(1.0, float(h)))

                    dist_m = round(1.8 / (norm_h + 1e-6), 2)
                    threat_val = max(0.0, min(1.0, float(scores[idx])))

                    targets.append(
                        ThreatBox(
                            id=f"vector_{track_id:02d}",
                            threat_score=threat_val,
                            distance_m=dist_m,
                            bbox=BoundaryBox(
                                x_min=x_min,
                                y_min=y_min,
                                width=norm_w,
                                height=norm_h,
                            ),
                        )
                    )

            payload = DataPayload(
                timestamp_ms=int(time.time() * 1000), targets=targets
            )

            await websocket.send_text(payload.model_dump_json())

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WebSocket Error ({user.id}): {e}")
    finally:
        receiver_task.cancel()