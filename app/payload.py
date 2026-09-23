import asyncio
import json
import logging
import math
import os
import time

import cv2
import numpy as np
import torch
from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect, status
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from ultralytics import YOLO

from app.db import db, supabase_admin
from app.sos import SOSFireError, fire_sos, spawn
from app.threat import ThreatEngine

log = logging.getLogger("suraksha.vision")
router = APIRouter(tags=["Vision HUD"])

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
POSE_WEIGHTS = os.getenv("POSE_WEIGHTS", "yolo11n-pose.pt")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")  
COOLDOWN_SECONDS = 5.0
VERDICT_TTL = 4.0            
GEMINI_LEVELS = ("MODERATE", "CRITICAL")   
MAX_GEMINI_INFLIGHT = 2    
MAX_FRAME_BYTES = 1_500_000
MAX_CONNECTIONS = int(os.getenv("MAX_WS_CONNECTIONS", "8"))
KP_CONF = 0.5               
SHOULDER_WIDTH_M = 0.40      
PERSON_HEIGHT_M = 1.65

# ---- auto-SOS ----
AUTO_SOS_ENABLED = os.getenv("AUTO_SOS_ENABLED", "1") == "1"
AUTO_SOS_COUNTDOWN_S = int(os.getenv("AUTO_SOS_COUNTDOWN_S", "8"))
AUTO_SOS_CONFIRMATIONS = int(os.getenv("AUTO_SOS_CONFIRMATIONS", "1"))  # verified hits needed…
AUTO_SOS_WINDOW_S = 15.0                                                # …within this window
AUTO_SOS_MAX_DIST_M = float(os.getenv("AUTO_SOS_MAX_DIST_M", "3.0"))    # only close threats arm
AUTO_SOS_SNOOZE_S = 60.0                                                # after user cancels
SNAPSHOT_BUCKET = os.getenv("SNAPSHOT_BUCKET", "sos-snapshots")

# COCO keypoint indices
NOSE, L_SH, R_SH, L_EL, R_EL, L_WR, R_WR, L_HIP, R_HIP = 0, 5, 6, 7, 8, 9, 10, 11, 12

SYSTEM_INSTRUCTION = (
    "You are a security assessment system for a personal-safety HUD. "
    "Judge ONLY visible body language in the crop: brandishing a weapon, throwing a punch, "
    "lunging, grabbing, raised fist aimed at the camera. "
    "Waving, stretching, dancing, phone use, sitting, carrying bags are NOT threats. "
    "When unsure, answer is_threat=false. If a threat is confirmed, give a short, calm, "
    "actionable instruction for the user."
)

_gemini = genai.Client()
_gpu_lock = asyncio.Semaphore(1 if DEVICE == "cuda" else 2)
_active_connections = 0

YOLO(POSE_WEIGHTS)


class BoundaryBox(BaseModel):
    x_min: float = Field(..., ge=0.0, le=1.0)
    y_min: float = Field(..., ge=0.0, le=1.0)
    width: float = Field(..., ge=0.0, le=1.0)
    height: float = Field(..., ge=0.0, le=1.0)


class ThreatBox(BaseModel):
    id: str
    level: str = "NORMAL"             
    threat_score: float = Field(..., ge=0.0, le=1.0)
    approach_mps: float = 0.0
    dwell_s: int = 0
    verified_threat: bool = False
    verification_pending: bool = False
    threat_reasoning: str = "Unverified"
    unity_instruction: str = ""
    distance_m: float
    bbox: BoundaryBox


class DataPayload(BaseModel):
    type: str = "frame"
    timestamp_ms: int
    frame_latency_ms: int
    targets: list[ThreatBox] = []


class GeminiVerdict(BaseModel):
    is_threat: bool
    reason: str
    unity_instruction: str


# ---------- math ----------
def _kp(kxy: np.ndarray, kconf: np.ndarray | None, i: int) -> np.ndarray | None:
    if kxy is None or i >= len(kxy):
        return None
    if kconf is not None and kconf[i] < KP_CONF:
        return None
    p = kxy[i]
    return None if (p[0] <= 0 and p[1] <= 0) else p


def estimate_distance(kxy, kconf, box_h_px: float, w_img: int, h_img: int, hfov_deg: float) -> float:
    fx = w_img / (2 * math.tan(math.radians(hfov_deg) / 2))
    vfov = 2 * math.atan(math.tan(math.radians(hfov_deg) / 2) * h_img / w_img)
    fy = h_img / (2 * math.tan(vfov / 2))
    ls, rs = _kp(kxy, kconf, L_SH), _kp(kxy, kconf, R_SH)
    if ls is not None and rs is not None:
        sh_px = float(np.linalg.norm(ls - rs))
        if sh_px > 8:  # too small = side-on / noise; fall through to height
            return float(np.clip(fx * SHOULDER_WIDTH_M / sh_px, 0.5, 20.0))
    return float(np.clip(fy * PERSON_HEIGHT_M / max(box_h_px, 1.0), 0.5, 20.0))


# ---------- gemini ----------
async def verify_with_gemini(crop: np.ndarray, dist_m: float) -> tuple[bool, str, str]:
    try:
        ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            return False, "Encode failed", ""
        resp = await asyncio.wait_for(
            _gemini.aio.models.generate_content(
                model=GEMINI_MODEL,
                contents=[
                    types.Part.from_bytes(data=buf.tobytes(), mime_type="image/jpeg"),
                    f"Person is ~{dist_m:.1f} m from the user. Immediate physical threat? "
                    "reason: under 10 words. unity_instruction: under 8 words, empty if no threat.",
                ],
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    response_mime_type="application/json",
                    response_schema=GeminiVerdict,
                    temperature=0.1,
                    max_output_tokens=120,
                ),
            ),
            timeout=4.0,
        )
        v: GeminiVerdict | None = resp.parsed
        if v is None:
            v = GeminiVerdict.model_validate_json(resp.text)
        return v.is_threat, v.reason[:80], (v.unity_instruction[:60] if v.is_threat else "")
    except asyncio.TimeoutError:
        return False, "Verification timeout", ""
    except Exception as e:
        log.warning("gemini verify error: %s", e)
        return False, "Verification failed", ""


async def upload_snapshot(user_id: str, frame: np.ndarray) -> str | None:
    try:
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            return None
        path = f"{user_id}/{int(time.time() * 1000)}.jpg"
        await db(lambda: supabase_admin.storage.from_(SNAPSHOT_BUCKET).upload(
            path, buf.tobytes(), {"content-type": "image/jpeg"}))
        return f"{SNAPSHOT_BUCKET}/{path}"
    except Exception as e:
        log.error("snapshot upload failed: %s", e)
        return None


@router.websocket("/prod/main")
async def main_suraksha(websocket: WebSocket, token: str = Query(None),
                        hfov: float = Query(65.0, ge=20.0, le=150.0)):
    global _active_connections

    if not token:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Missing token")
        return
    try:
        res = await asyncio.to_thread(supabase_admin.auth.get_user, token)
        user = res.user if res else None
    except Exception:
        user = None
    if not user:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Invalid token")
        return
    if _active_connections >= MAX_CONNECTIONS:
        await websocket.close(code=status.WS_1013_TRY_AGAIN_LATER, reason="Server busy")
        return

    await websocket.accept()
    _active_connections += 1

    # ---- per-connection state (nothing shared between users) ----
    model = YOLO(POSE_WEIGHTS)  # own instance = own ByteTrack state (nano ≈ 6 MB, loads in ms)
    cooldown: dict[str, float] = {}
    pending: dict[str, asyncio.Task] = {}
    verdicts: dict[str, tuple[float, bool, str, str]] = {}  # key → (ts, is_threat, reason, cmd)
    engine = ThreatEngine()

    latest: tuple[bytes, float] | None = None
    new_frame = asyncio.Event()
    closed = False
    send_lock = asyncio.Lock()

    # auto-SOS state
    gps: dict = {}                       # last {"latitude","longitude","accuracy_m","speed_mps","battery_percentage","state_code","district"}
    confirmations: list[float] = []      # timestamps of close, verified threats
    countdown_task: asyncio.Task | None = None
    armed_frame: np.ndarray | None = None
    armed_reason = ""
    snooze_until = 0.0
    fired_incident: dict | None = None
    firing: asyncio.Task | None = None   # shielded so a dropped socket can't kill an in-flight SOS

    async def send_json(obj: dict | BaseModel):
        text = obj.model_dump_json() if isinstance(obj, BaseModel) else json.dumps(obj)
        async with send_lock:
            await websocket.send_text(text)

    async def do_fire(frame_img: np.ndarray | None, reason: str) -> dict | None:
        nonlocal fired_incident
        snap = await upload_snapshot(user.id, frame_img) if frame_img is not None else None
        try:
            fired_incident = await fire_sos(
                user,
                latitude=gps.get("latitude"), longitude=gps.get("longitude"),
                accuracy_m=gps.get("accuracy_m", 0.0), speed_mps=gps.get("speed_mps", 0.0),
                battery_percentage=gps.get("battery_percentage"),
                state_code=gps.get("state_code", "NATIONAL"), district=gps.get("district", "All"),
                snapshot_url=snap, threat_level="CRITICAL", source="AUTO_HUD",
            )
            log.warning("AUTO SOS user=%s reason=%s", user.id, reason)
            return fired_incident
        except SOSFireError as e:
            log.error("auto SOS failed: %s", e.detail)
            return None

    def start_fire(reason: str) -> asyncio.Task:
        nonlocal firing
        if firing is None:
            firing = spawn(do_fire(armed_frame, reason))
        return firing

    async def countdown():
        nonlocal countdown_task
        try:
            for left in range(AUTO_SOS_COUNTDOWN_S, 0, -1):
                await send_json({"type": "sos_countdown", "seconds_left": left, "reason": armed_reason,
                                 "unity_instruction": f"Sending SOS in {left}. Say cancel if safe."
                                 if left == AUTO_SOS_COUNTDOWN_S else ""})
                await asyncio.sleep(1)
            inc = await asyncio.shield(start_fire(armed_reason))
            countdown_task = None
            await send_json({"type": "sos_fired", "incident": inc,
                             "unity_instruction": "SOS sent. Help is being alerted."} if inc else
                            {"type": "sos_failed", "client_should_dial": "112",
                             "unity_instruction": "SOS failed. Dial 112 now."})
        except asyncio.CancelledError:
            raise
        except Exception as e:
            # socket died mid-countdown → finally-block dead-man switch handles it
            log.warning("countdown interrupted: %s", e)

    def maybe_arm(now: float, frame_img: np.ndarray, reason: str):
        nonlocal countdown_task, armed_frame, armed_reason
        if not AUTO_SOS_ENABLED or countdown_task or firing or now < snooze_until:
            return
        confirmations[:] = [t for t in confirmations if now - t <= AUTO_SOS_WINDOW_S] + [now]
        if len(confirmations) >= AUTO_SOS_CONFIRMATIONS:
            armed_frame, armed_reason = frame_img.copy(), reason
            countdown_task = asyncio.create_task(countdown())

    async def handle_control(msg: dict):
        nonlocal countdown_task, snooze_until, fired_incident
        kind = msg.get("type")
        if kind == "gps":
            for k in ("latitude", "longitude", "accuracy_m", "speed_mps"):
                if isinstance(msg.get(k), (int, float)):
                    gps[k] = float(msg[k])
            if isinstance(msg.get("speed_mps"), (int, float)):
                engine.set_user_speed(float(msg["speed_mps"]))
            if isinstance(msg.get("battery_percentage"), int):
                gps["battery_percentage"] = max(0, min(100, msg["battery_percentage"]))
            for k in ("state_code", "district"):
                if isinstance(msg.get(k), str):
                    gps[k] = msg[k][:64]
        elif kind == "audio":   # phone-side scream/shout classifier (YAMNet) → {"type":"audio","scream":0.0-1.0}
            if isinstance(msg.get("scream"), (int, float)):
                engine.set_audio(float(msg["scream"]))
        elif kind == "cancel_sos":  # voice "cancel" / tap on glasses → only cancels a countdown
            if countdown_task and not countdown_task.done() and firing is None:
                countdown_task.cancel()
                countdown_task = None
                confirmations.clear()
                snooze_until = time.time() + AUTO_SOS_SNOOZE_S
                await send_json({"type": "sos_cancelled", "snooze_s": AUTO_SOS_SNOOZE_S,
                                 "unity_instruction": "SOS cancelled."})
        elif kind == "manual_sos":  # panic gesture / voice "help" → skip countdown
            if countdown_task and not countdown_task.done():
                countdown_task.cancel()
                countdown_task = None
            inc = await asyncio.shield(start_fire("Manual trigger from glasses"))
            await send_json({"type": "sos_fired", "incident": inc} if inc else
                            {"type": "sos_failed", "client_should_dial": "112"})

    def infer(frame: np.ndarray):
        with torch.inference_mode():
            return model.track(source=frame, persist=True, tracker="bytetrack.yaml", classes=[0],
                               imgsz=320, device=DEVICE, verbose=False, conf=0.4)

    async def receiver():
        nonlocal latest, closed
        try:
            while True:
                msg = await websocket.receive()
                if msg["type"] == "websocket.disconnect":
                    break
                if msg.get("bytes") is not None:
                    if len(msg["bytes"]) <= MAX_FRAME_BYTES:
                        latest = (msg["bytes"], time.time())
                        new_frame.set()
                elif msg.get("text"):
                    try:
                        await handle_control(json.loads(msg["text"]))
                    except (ValueError, TypeError):
                        pass
        except Exception:
            pass
        finally:
            closed = True
            new_frame.set()

    recv_task = asyncio.create_task(receiver())

    try:
        while True:
            await new_frame.wait()
            new_frame.clear()
            if closed:
                break
            if latest is None:
                continue
            jpeg, recv_ts = latest
            latest = None

            frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue
            h_img, w_img = frame.shape[:2]

            async with _gpu_lock:
                results = await asyncio.to_thread(infer, frame)

            now = time.time()
            targets: list[ThreatBox] = []
            r0 = results[0] if results else None

            if r0 is not None and r0.boxes is not None and len(r0.boxes) > 0:
                xyxy = r0.boxes.xyxy.cpu().numpy()
                ids = r0.boxes.id.cpu().numpy().astype(int) if r0.boxes.id is not None else None
                kxy_all = r0.keypoints.xy.cpu().numpy() if r0.keypoints is not None else None
                kcf_all = (r0.keypoints.conf.cpu().numpy()
                           if r0.keypoints is not None and r0.keypoints.conf is not None else None)

                for i in range(len(xyxy)):
                    if ids is None:
                        continue  # untracked detection: no stable id → skip until tracker locks on
                    key = f"vector_{ids[i]:02d}"
                    x1, y1, x2, y2 = xyxy[i]
                    x1, y1 = max(0.0, float(x1)), max(0.0, float(y1))
                    x2, y2 = min(float(w_img), float(x2)), min(float(h_img), float(y2))
                    bw, bh = x2 - x1, y2 - y1
                    if bw <= 1 or bh <= 1:
                        continue

                    kxy = kxy_all[i] if kxy_all is not None and i < len(kxy_all) else None
                    kcf = kcf_all[i] if kcf_all is not None and i < len(kcf_all) else None

                    dist = round(estimate_distance(kxy, kcf, bh, w_img, h_img, hfov), 2)
                    cx_norm = (x1 + bw / 2) / w_img
                    assess = engine.update(key, now, dist, cx_norm, bw, bh, kxy, kcf, KP_CONF)
                    score, level = assess["score"], assess["level"]

                    # collect finished Gemini results
                    t = pending.get(key)
                    if t is not None and t.done():
                        pending.pop(key)
                        try:
                            ok, why, cmd = t.result()
                        except Exception:
                            ok, why, cmd = False, "Verification failed", ""
                        verdicts[key] = (now, ok, why, cmd)
                        if ok and level == "CRITICAL" and dist <= AUTO_SOS_MAX_DIST_M:
                            maybe_arm(now, frame, why)

                    # fire a new verification (non-blocking)
                    if (level in GEMINI_LEVELS and key not in pending
                            and len(pending) < MAX_GEMINI_INFLIGHT
                            and now - cooldown.get(key, 0) > COOLDOWN_SECONDS):
                        pad_x, pad_y = bw * 0.15, bh * 0.10
                        cx1, cy1 = int(max(0, x1 - pad_x)), int(max(0, y1 - pad_y))
                        cx2, cy2 = int(min(w_img, x2 + pad_x)), int(min(h_img, y2 + pad_y))
                        crop = frame[cy1:cy2, cx1:cx2]
                        if crop.size > 0:
                            pending[key] = asyncio.create_task(verify_with_gemini(crop.copy(), dist))
                            cooldown[key] = now

                    v = verdicts.get(key)
                    if v and now - v[0] > VERDICT_TTL:
                        verdicts.pop(key, None)
                        v = None
                    verified, reason, cmd = (v[1], v[2], v[3]) if v else (
                        False, assess["reason"], "")

                    targets.append(ThreatBox(
                        id=key,
                        level=level,
                        threat_score=score,
                        approach_mps=assess["approach_mps"],
                        dwell_s=assess["dwell_s"],
                        verified_threat=verified,
                        verification_pending=key in pending,
                        threat_reasoning=reason,
                        unity_instruction=cmd,
                        distance_m=dist,
                        bbox=BoundaryBox(
                            x_min=x1 / w_img, y_min=y1 / h_img,
                            width=min(1.0, bw / w_img), height=min(1.0, bh / h_img),
                        ),
                    ))

            # forget tracks that disappeared
            live = {t.id for t in targets}
            engine.drop_missing(live, now)
            for k in list(verdicts):
                if k not in live and k not in pending:
                    verdicts.pop(k, None)

            payload = DataPayload(timestamp_ms=int(now * 1000),
                                  frame_latency_ms=int((time.time() - recv_ts) * 1000),
                                  targets=targets)
            await send_json(payload)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        log.exception("WebSocket error (%s): %s", user.id, e)
    finally:
        _active_connections -= 1
        recv_task.cancel()
        for t in pending.values():
            t.cancel()

        if countdown_task and not countdown_task.done():
            countdown_task.cancel()
            if firing is None:
                log.warning("dead-man switch: firing SOS for %s", user.id)
                start_fire(f"Connection lost during threat: {armed_reason}")
        try:
            await websocket.close()
        except Exception:
            pass
