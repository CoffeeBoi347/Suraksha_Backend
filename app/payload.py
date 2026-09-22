import asyncio
import json
import os
import time
import cv2
import numpy as np
import torch
from dotenv import load_dotenv
from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect, status
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from supabase import Client, create_client
from ultralytics import YOLO

load_dotenv()

SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

gemini_client = genai.Client()

app = FastAPI(title="Suraksha Core Engine")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO("yolo11n-pose.pt")

THREAT_ALERT_COOLDOWN: dict[str, float] = {}
COOLDOWN_SECONDS = 5.0 
GEMINI_MODEL = "gemini-3.5-flash-lite"
SYSTEM_INSTRUCTION = (
    "You are an AI Security Assessment system operating in a real-time HUD environment. "
    "Analyze security frames and visual crops for aggressive or hostile body language "
    "(e.g., brandishing weapon, throwing punches, lunging). Distinguish aggressive actions "
    "from harmless movements (stretching, waving, sitting). "
    "If a threat is confirmed, output a brief, direct, tactical audio instruction for the user."
)


class BoundaryBox(BaseModel):
    x_min: float = Field(..., ge=0.0, le=1.0)
    y_min: float = Field(..., ge=0.0, le=1.0)
    width: float = Field(..., ge=0.0, le=1.0)
    height: float = Field(..., ge=0.0, le=1.0)


class ThreatBox(BaseModel):
    id: str
    threat_score: float = Field(..., ge=0.0, le=1.0)
    verified_threat: bool = False
    threat_reasoning: str = "Unverified"
    unity_instruction: str = ""
    distance_m: float
    bbox: BoundaryBox


class DataPayload(BaseModel):
    timestamp_ms: int
    targets: list[ThreatBox] = []


def calculate_pose_threat(keypoints_xy: np.ndarray, bbox_w: float, bbox_h: float, distance_m: float) -> float:
    pose_threat = 0.0

    if keypoints_xy is not None and len(keypoints_xy) >= 11:
        l_shoulder = keypoints_xy[5]
        r_shoulder = keypoints_xy[6]
        l_wrist = keypoints_xy[9]
        r_wrist = keypoints_xy[10]

        if l_wrist[1] > 0 and r_wrist[1] > 0:
            if l_wrist[1] < l_shoulder[1] or r_wrist[1] < r_shoulder[1]:
                pose_threat += 0.35
        if bbox_w > (bbox_h * 1.2):
            pose_threat += 0.40

    proximity_factor = max(0.0, min(1.0, (5.0 - distance_m) / 5.0)) * 0.25
    total_threat = pose_threat + proximity_factor
    return float(np.clip(total_threat, 0.05, 1.0))


async def verify_threat_with_gemini(crop_img: np.ndarray, distance_m: float) -> tuple[bool, str, str]:
    """
    Sends visual crop to Gemini 3.5 Flash Lite for threat verification.
    """
    try:
        _, buffer = cv2.imencode(".jpg", crop_img)
        jpeg_bytes = buffer.tobytes()

        user_prompt = (
            f"Target distance: {distance_m} meters. "
            "Analyze if this person poses an immediate physical threat. "
            "Respond ONLY in valid JSON with these keys:\n"
            "{\n"
            '  "is_threat": boolean,\n'
            '  "reason": "Short visual diagnosis under 10 words",\n'
            '  "unity_instruction": "Short tactical command for audio alert under 8 words (e.g., \'Hostile action detected, step back\')"\n'
            "}"
        )

        response = await gemini_client.aio.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                types.Part.from_bytes(data=jpeg_bytes, mime_type="image/jpeg"),
                user_prompt,
            ],
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                response_mime_type="application/json",
                temperature=0.1,
                thinking_config=types.ThinkingConfig(thinking_budget=0), 
            ),
        )

        result = json.loads(response.text)
        return (
            result.get("is_threat", False),
            result.get("reason", "No reason provided."),
            result.get("unity_instruction", "") if result.get("is_threat") else "",
        )

    except Exception as e:
        print(f"Gemini crop verification error: {e}")
        return False, "Verification failed", ""


async def analyze_single_frame_threat(frame_img: np.ndarray) -> dict:
    """
    Analyzes an entire single frame directly with Gemini 3.5 Flash Lite for threats.
    """
    try:
        _, buffer = cv2.imencode(".jpg", frame_img)
        jpeg_bytes = buffer.tobytes()

        user_prompt = (
            "Analyze this video frame for threatening behavior or aggression. "
            "Respond ONLY in valid JSON with these keys:\n"
            "{\n"
            '  "is_threat": boolean,\n'
            '  "reason": "Brief visual description under 12 words",\n'
            '  "unity_instruction": "Tactical HUD audio alert under 8 words"\n'
            "}"
        )

        response = await gemini_client.aio.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                types.Part.from_bytes(data=jpeg_bytes, mime_type="image/jpeg"),
                user_prompt,
            ],
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                response_mime_type="application/json",
                temperature=0.1,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )

        return json.loads(response.text)

    except Exception as e:
        print(f"Gemini full frame analysis error: {e}")
        return {"is_threat": False, "reason": "Frame evaluation failed", "unity_instruction": ""}


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

            h_img, w_img, _ = frame.shape
            results = await asyncio.to_thread(run_inference, frame)
            targets: list[ThreatBox] = []
            now = time.time()

            if results and len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
                boxes_obj = results[0].boxes
                boxes = boxes_obj.xywhn.cpu().numpy()

                track_ids = (
                    boxes_obj.id.cpu().numpy().astype(int)
                    if boxes_obj.id is not None
                    else list(range(len(boxes)))
                )

                keypoints_data = None
                if results[0].keypoints is not None:
                    keypoints_data = results[0].keypoints.xy.cpu().numpy()

                for idx, track_id in enumerate(track_ids):
                    target_key = f"vector_{track_id:02d}"
                    cx, cy, w, h = boxes[idx]

                    x_min = max(0.0, min(1.0, float(cx - (w / 2.0))))
                    y_min = max(0.0, min(1.0, float(cy - (h / 2.0))))
                    norm_w = max(0.0, min(1.0, float(w)))
                    norm_h = max(0.0, min(1.0, float(h)))

                    kpts = keypoints_data[idx] if keypoints_data is not None and idx < len(keypoints_data) else None
                    if kpts is not None and len(kpts) >= 7 and kpts[5][0] > 0 and kpts[6][0] > 0:
                        shoulder_px_w = abs(kpts[5][0] - kpts[6][0]) / w_img
                        dist_m = round(0.45 / (shoulder_px_w + 1e-6), 2)
                    else:
                        dist_m = round(1.75 / (norm_h + 1e-6), 2)

                    dist_m = max(0.5, min(20.0, dist_m))
                    threat_val = calculate_pose_threat(kpts, norm_w, norm_h, dist_m)

                    is_verified = False
                    reason = "Low threat score"
                    unity_cmd = ""

                    last_alert_time = THREAT_ALERT_COOLDOWN.get(target_key, 0)
                    can_trigger_gemini = (now - last_alert_time) > COOLDOWN_SECONDS

                    if threat_val >= 0.60 and can_trigger_gemini:
                        px_xmin = int(x_min * w_img)
                        px_ymin = int(y_min * h_img)
                        px_xmax = int((x_min + norm_w) * w_img)
                        px_ymax = int((y_min + norm_h) * h_img)

                        crop = frame[px_ymin:px_ymax, px_xmin:px_xmax]
                        if crop.size > 0:
                            is_verified, reason, unity_cmd = await verify_threat_with_gemini(crop, dist_m)
                            THREAT_ALERT_COOLDOWN[target_key] = now

                    targets.append(
                        ThreatBox(
                            id=target_key,
                            threat_score=threat_val,
                            verified_threat=is_verified,
                            threat_reasoning=reason,
                            unity_instruction=unity_cmd,
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