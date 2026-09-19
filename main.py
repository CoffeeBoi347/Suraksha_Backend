from pydantic import Field, BaseModel
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from ultralytics import YOLO
import cv2
import numpy as np
import time

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

@app.websocket("/prod/main")
async def main_suraksha(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            jpeg_bytes = websocket.receive_bytes()
            np_arr = np.frombuffer(jpeg_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if frame is None:
                continue

            results = model.track(
                source=frame,
                persist=True,
                tracker="bytetracker.yaml",
                classes=[0],
                verbose=False
            )

            targets: list[ThreatBox] = []
            if results and results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xywhn.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            else:
                track_ids = []

                for idx, track_id in enumerate(track_ids):
                    cx,cy,w,h = boxes[idx]

                    x_min = max(0.0, min(1.0, float(cx - (w / 2.0))))
                    y_min = max(0.0, min(1.0, float(cy - (h / 2.0))))
                    norm_w = max(0.0, min(1.0, float(w)))
                    norm_h = max(0.0, min(1.0, float(h)))

                    dist_m = round(1.8 / (norm_h + 1e-6), 2)

                    threat = ThreatBox(
                        id=f"vector_{track_id:02d}",
                        threat_score=0.0,
                        distance_m=dist_m,
                        bbox=BoundaryBox(
                            x_min=x_min,
                            y_min=y_min,
                            width=norm_w,
                            height=norm_h
                        )
                    )

                    targets.append(threat)

            payload = DataPayload(
                timestamp_ms=int(time.time() * 1000),
                targets=targets
            )

            await websocket.send_text(payload.model_dump_json()) # Returns JSON

    except WebSocketDisconnect:
        print("Backend Client Disconnected.")