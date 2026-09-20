import asyncio
import time
import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field
from ultralytics import YOLO

app = FastAPI(title="Suraksha Backend")
model = YOLO("yolo11n-pose.pt")


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
    """Synchronous YOLO tracking function to offload from FastAPI main event loop."""
    return model.track(
        source=frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=[0],
        verbose=False,
    )


@app.websocket("/prod/main")
async def main_suraksha(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            jpeg_bytes = await websocket.receive_bytes()
            np_arr = np.frombuffer(jpeg_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if frame is None:
                continue

            results = await asyncio.to_thread(run_inference, frame)

            targets: list[ThreatBox] = []

            if (
                results
                and len(results) > 0
                and results[0].boxes is not None
                and len(results[0].boxes) > 0
            ):
                boxes_obj = results[0].boxes
                boxes = boxes_obj.xywhn.cpu().numpy()

                if boxes_obj.id is not None:
                    track_ids = boxes_obj.id.cpu().numpy().astype(int)
                else:
                    track_ids = list(range(len(boxes)))

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

                    threat = ThreatBox(
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
                    targets.append(threat)

            payload = DataPayload(
                timestamp_ms=int(time.time() * 1000), targets=targets
            )

            await websocket.send_text(payload.model_dump_json())

    except WebSocketDisconnect:
        print("Backend Client Disconnected.")
    except Exception as e:
        print(f"WebSocket Error: {e}")