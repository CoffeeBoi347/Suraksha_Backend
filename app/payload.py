"""Suraksha vision HUD: pose + tracking + objects -> behaviour cues -> ThreatEngine/fuse -> Gemini check -> HUD / SOS.

Accuracy changes vs previous version (cue logic lives in app/cues.py, unit-tested):
 1. Motion is timed by frame capture/arrival, not by when inference finished (object-pass frames faked fast punches).
 2. Distance = smallest sane of shoulder / torso / height estimates + 3-frame median (side-on people read 5x too far).
 3. Pose cues use a side-on-robust body scale; "reach" needs a straight arm at chest height (old rule fired on
    hanging arms, which cascaded into object_swing for anyone side-on holding a bottle).
 4. Holding a bottle / umbrella / book is not a cue; raised, swung, or a blade in hand is. Per-label debounce,
    person-crop pass for small blades (GPU), vehicles and two-wheeler riders / pillion detected.
 5. Pre-attack indicators: target glancing, scanning for witnesses, re-approach, two-wheeler pair lingering.
 6. Following = TEDD stop/turn tests or pacing alongside / looking back, not "anyone near for 20 s".
 7. Crowd baseline: when most people move their arms alike (dance, procession), arm motion beyond 2 m is discounted.
 8. Gemini sees the scene with the target boxed + the trigger-moment crop; scans need 0.75 conf; a confident
    "none" clears visual-only MODERATE; calls are prioritised across tracks.
 9. Safe zones no longer silence strikes / weapons (most violence against women in India is domestic).
10. Company mode, HUD hysteresis, MODERATE events + cue features logged for calibration (app/calibrate.py).
11. Counter-reasoning gate (app/intent.py): every person is first read as away / busy / passing / neutral /
    watching / targeted. Approach / social cues only count for "targeted"; arm motion only counts when aimed at
    her. Local cues alone can no longer show CRITICAL.
12. Gemini is not told the detector's conclusion and must state the benign explanation first.
13. SOS popup: every CRITICAL within AUTO_SOS_MAX_DIST_M opens it (see AUTO_SOS_* below). Countdown runs in the app.
    Server -> client: sos_prompt, then sos_fired / sos_failed / sos_cancelled / sos_already_sent.
    Client -> server: confirm_sos, cancel_sos. Server fires itself after seconds + AUTO_SOS_GRACE_S of silence.
14. AUTO_SOS_ON_CRITICAL (default on): 0 = only prompt on Gemini-verified SOS-grade threats.
15. Gemini cost control: CLEARED_HOLD_S, SCAN_GAP_S, GEMINI_USER_RPM.

SPEED (this version):
 P1. Two-stage pipeline per connection: stage 1 (decode + pose/track on GPU, in threads) runs while stage 2
     (cues, fusion, HUD) processes the previous frame. GPU and CPU work overlap instead of alternating.
 P2. Object/vehicle pass runs in its own background task on the newest frame (time-gated by OBJ_MIN_GAP_S),
     so it no longer stalls every 2nd/3rd frame (that was the stutter).
 P3. Big JPEGs decode straight at 1/2, 1/4 or 1/8 size (PROC_MAX_SIDE); far cheaper than decode + resize.
 P4. FP16 on CUDA (YOLO_HALF), GPU->CPU copies happen in the worker thread, not on the event loop.
 P5. Re-ID embeddings batched in a thread and refreshed per track every REID_EVERY_S, not every frame.
 P6. Scene context every SCENE_EVERY_S; audio classification, JPEG encoding, bystander blur and model loading
     all moved off the event loop (they were blocking every connection).
 P7. HUD boxes carry vx / vy (normalised units per second): the client extrapolates between packets, so the
     overlay moves at display rate (60/90 Hz) even when inference runs at 15-25 fps.
 P8. Gemini: config built once; optional GEMINI_MEDIA_RES=low|medium and GEMINI_THINKING_BUDGET cut tokens.
Optional client upgrades: prefix each JPEG with b"SRK1" + int64 little-endian capture time in ms; add heading_deg to gps.
"""
import asyncio
import base64
from collections import deque
from enum import Enum
import json
import logging
import os
import re
import time
from datetime import datetime, timezone

import cv2
import numpy as np
import torch
from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect, status
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from ultralytics import YOLO

from app.cues import (
    HAND_OBJECTS, KP_CONF, OBJECT_CLASSES, SHARP, TWO_WHEELERS, VEHICLES,
    ApproachEpisodes, CrowdBaseline, FollowTest, FrameClock, GazeTracker, Median3,
    classify_speech, estimate_distance, head_state, match_object, merge_objects, parse_objects,
    pose_signals, pre_attack_cues, ride_flags,
)
from app.db import db, supabase_admin
from app.sos import SOSFireError, fire_sos, spawn
from app.reid import PersonGallery, embed_person
from app.faceblur import blur_bystanders
from app.audio_classify import get_classifier as get_audio_classifier
from app.scene import SceneContext
from app.threat import ThreatEngine, fuse
from app.intent import IntentTracker, locate

__all__ = ["router", "classify_speech", "estimate_distance", "pose_signals", "verify_with_gemini"]

log = logging.getLogger("suraksha.vision")
logging.getLogger("google_genai.models").setLevel(logging.WARNING)
router = APIRouter(tags=["Vision HUD"])

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
POSE_WEIGHTS = os.getenv("POSE_WEIGHTS", "yolo11n-pose.pt")   # .engine (TensorRT) works too, see notes
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

# ---- speed ----
HALF = DEVICE == "cuda" and os.getenv("YOLO_HALF", "1") == "1"
PROC_MAX_SIDE = int(os.getenv("PROC_MAX_SIDE", "960"))        # decode bigger frames at 1/2, 1/4, 1/8 (0 = off)
OBJ_MIN_GAP_S = float(os.getenv("OBJ_MIN_GAP_S", "0.12" if DEVICE == "cuda" else "0.3"))
REID_EVERY_S = float(os.getenv("REID_EVERY_S", "0.4"))
SCENE_EVERY_S = float(os.getenv("SCENE_EVERY_S", "0.25"))
VEL_EMA = 0.5
VEL_MAX = 3.0                     # normalised screen units / s
cv2.setNumThreads(int(os.getenv("CV_THREADS", "2")))           # many to_thread workers: stop OpenCV oversubscribing
if DEVICE == "cpu":
    torch.set_num_threads(int(os.getenv("TORCH_THREADS", str(max(1, (os.cpu_count() or 4) // 2)))))

# ---- people / behaviour ----
YOLO_IMGSZ = int(os.getenv("YOLO_IMGSZ", "480" if DEVICE == "cuda" else "320"))
YOLO_CONF = float(os.getenv("YOLO_CONF", "0.25"))
FOLLOW_SECONDS = float(os.getenv("FOLLOW_SECONDS", "20"))
FOLLOW_MAX_DIST_M = float(os.getenv("FOLLOW_MAX_DIST_M", "6.0"))
FOLLOW_MIN_USER_SPEED = 0.6
ALONGSIDE_MAX_DIST_M = 4.0
ALONGSIDE_EDGE = 0.2
ALONGSIDE_FRAC = 0.6
CROWD_DISCOUNT_MIN_DIST_M = 2.0
REID_MIN_BOX_H = int(os.getenv("REID_MIN_BOX_H", "48"))
TRACK_TTL_S = 8.0
SOCIAL_GATED = {"converge", "converge_fast", "flank", "engage", "block", "stare", "linger", "ambush",
                "encircle", "rejected_follow", "verbal_abuse", "verbal_threat"}
PHYSICAL_ACT = {"strike", "contact", "object_swing"}
DIRECT_DIST_M = 1.5
WEAPON_DIRECT_DIST_M = 2.0
LOCAL_CUE_STATES = {"targeted", "watching"}
SOCIAL_CATS = {"follow", "follow_strong", "linger", "stare", "engage", "rejected_follow"}
HARD_AT_HOME = {"strike", "weapon_held", "object_swing"}

# ---- gemini ----
COOLDOWN_CRITICAL_S = 4.0
COOLDOWN_MODERATE_S = 8.0
COOLDOWN_SCAN_S = 10.0
GEMINI_RPM = int(os.getenv("GEMINI_RPM", "10"))
GEMINI_RANK1_BUDGET = 0.6
GEMINI_SCAN_BUDGET = 0.3
GEMINI_SCAN_DIST_M = 4.0
CLEARED_HOLD_S = float(os.getenv("CLEARED_HOLD_S", "30"))
SCAN_GAP_S = float(os.getenv("SCAN_GAP_S", "15"))
GEMINI_USER_RPM = int(os.getenv("GEMINI_USER_RPM", "6"))
GEMINI_ENABLED = os.getenv("GEMINI_ENABLED", "1") == "1"        # kill switch: 0 = local detection only
GEMINI_USER_HOURLY = int(os.getenv("GEMINI_USER_HOURLY", "60"))
GEMINI_USER_DAILY = int(os.getenv("GEMINI_USER_DAILY", "250"))
GEMINI_GLOBAL_DAILY = int(os.getenv("GEMINI_GLOBAL_DAILY", "1500"))
CONTROL_MSGS_PER_S = int(os.getenv("CONTROL_MSGS_PER_S", "30"))
MAX_SESSION_S = float(os.getenv("MAX_SESSION_S", "0"))           # 0 = off, e.g. 14400 for 4 h
MANUAL_MIN_GAP_S = float(os.getenv("MANUAL_MIN_GAP_S", "5"))
DEMO_LAT = os.getenv("DEMO_LAT")                                 # demo only: fixed location when the client sends no GPS
DEMO_LNG = os.getenv("DEMO_LNG")
VERDICT_TTL = 6.0
GEMINI_LEVELS = ("MODERATE", "MEDIUM", "CRITICAL")
GEMINI_MIN_CONF = float(os.getenv("GEMINI_MIN_CONF", "0.7"))
GEMINI_SCAN_MIN_CONF = float(os.getenv("GEMINI_SCAN_MIN_CONF", "0.75"))
GEMINI_SUSPICIOUS_CONF = float(os.getenv("GEMINI_SUSPICIOUS_CONF", "0.65"))
GEMINI_CAN_CLEAR = os.getenv("GEMINI_CAN_CLEAR", "1") == "1"
GEMINI_CLEAR_CONF = float(os.getenv("GEMINI_CLEAR_CONF", "0.75"))
GEMINI_RANK1_KEYS = {"strike", "contact", "pose", "converge_fast", "weapon_held", "object_held",
                     "object_swing", "flank", "ambush", "engage", "block", "follow_strong",
                     "linger", "rejected_follow", "stare", "verbal_threat"}
MAX_GEMINI_INFLIGHT = 2
CROP_JPEG_Q = int(os.getenv("CROP_JPEG_Q", "75"))
CONTEXT_JPEG_Q = int(os.getenv("CONTEXT_JPEG_Q", "65"))
_gemini_global = asyncio.Semaphore(int(os.getenv("GEMINI_GLOBAL_INFLIGHT", "6")))
_gemini_calls: deque = deque()
_gemini_day: deque = deque()
_user_gemini: dict[str, deque] = {}
_gemini_block_until = 0.0

TEMPORAL_FRAME_COUNT = int(os.getenv("TEMPORAL_FRAME_COUNT", "4"))
TEMPORAL_WINDOW_S = float(os.getenv("TEMPORAL_WINDOW_S", "2.0"))
TEMPORAL_MIN_FRAMES = int(os.getenv("TEMPORAL_MIN_FRAMES", "3"))
CROP_INTERVAL_S = TEMPORAL_WINDOW_S / TEMPORAL_FRAME_COUNT
CROP_MAX_DIST_M = 8.0
CROP_MAX_SIDE = 384
CONTEXT_MAX_SIDE = 512
TRIGGER_CROP_AGE_S = 0.15

# ---- objects / vehicles (second COCO pass, background task) ----
OBJ_DETECT = os.getenv("WEAPON_DETECT", "1") == "1"
OBJ_WEIGHTS = os.getenv("WEAPON_WEIGHTS", "yolo11n.pt")
OBJ_FLOOR_CONF = float(os.getenv("WEAPON_CONF", "0.25"))
OBJ_IMGSZ = int(os.getenv("WEAPON_IMGSZ", "640" if DEVICE == "cuda" else "480"))
OBJ_CROP_PASS = os.getenv("OBJ_CROP_PASS", "1" if DEVICE == "cuda" else "0") == "1"
OBJ_CROP_IMGSZ = int(os.getenv("OBJ_CROP_IMGSZ", "320"))
OBJ_CROP_MAX_PEOPLE = 2
OBJ_CROP_MAX_DIST_M = 4.0
WEAPON_MIN_HITS = int(os.getenv("WEAPON_MIN_HITS", "2"))
WEAPON_HIT_WINDOW_S = 2.5
OBJ_CACHE_TTL_S = 1.0
VEHICLE_FRESH_S = 0.35
_ALL_OBJ_IDS = list(OBJECT_CLASSES)
_HAND_OBJ_IDS = [i for i, n in OBJECT_CLASSES.items() if n in HAND_OBJECTS]
_MOVING = TWO_WHEELERS | VEHICLES

# ---- feedback / HUD / logging ----
FEEDBACK_WINDOW_S = 120.0
FEEDBACK_LABELS = ("false_alarm", "real_threat", "unsure")
MAX_UI_TARGETS = int(os.getenv("MAX_UI_TARGETS", "8"))
SEND_NORMAL_BOXES = os.getenv("SEND_NORMAL_BOXES", "1") == "1"
HUD_NORMAL_MAX_DIST_M = float(os.getenv("HUD_NORMAL_MAX_DIST_M", "8"))
MAX_FRAME_BYTES = 1_500_000
MAX_CONNECTIONS = int(os.getenv("MAX_WS_CONNECTIONS", "8"))
FRAME_MAGIC = b"SRK1"
HOLD_MODERATE_S = 2.5
HOLD_CRITICAL_S = 4.0
THREAT_LOG_COOLDOWN_S = 10.0
LOG_MODERATE = os.getenv("LOG_MODERATE", "1") == "1"
MODERATE_LOG_COOLDOWN_S = 30.0

# ---- SOS popup ----
AUTO_SOS_ENABLED = os.getenv("AUTO_SOS_ENABLED", "1") == "1"
AUTO_SOS_COUNTDOWN_S = int(os.getenv("AUTO_SOS_COUNTDOWN_S", "10"))
AUTO_SOS_GRACE_S = int(os.getenv("AUTO_SOS_GRACE_S", "5"))
AUTO_SOS_CONFIRMATIONS = int(os.getenv("AUTO_SOS_CONFIRMATIONS", "2"))
AUTO_SOS_MIN_CONF = float(os.getenv("AUTO_SOS_MIN_CONF", "0.85"))
AUTO_SOS_ON_CRITICAL = os.getenv("AUTO_SOS_ON_CRITICAL", "1") == "1"
AUTO_SOS_CRIT_SPACING_S = float(os.getenv("AUTO_SOS_CRIT_SPACING_S", "0.5"))
AUTO_SOS_REARM_S = float(os.getenv("AUTO_SOS_REARM_S", "30"))
SOS_PATTERNS = {"weapon", "strike", "grab_snatch", "restrain", "charge"}
SOS_THUMB_SIDE = 256
AUTO_SOS_SCREAM_CONFIRMATIONS = 1
AUTO_SOS_SCREAM_MIN = 0.7
AUTO_SOS_SCREAM_FRESH_S = 5.0
AUDIO_ALERT_MIN = 0.8
AUDIO_ALERT_HITS = 2
AUDIO_ALERT_WINDOW_S = 4.0
AUDIO_ALERT_REPEAT_S = 10.0
AUTO_SOS_WINDOW_S = float(os.getenv("AUTO_SOS_WINDOW_S", "20"))
AUTO_SOS_MAX_DIST_M = float(os.getenv("AUTO_SOS_MAX_DIST_M", "3.0"))
AUTO_SOS_SNOOZE_S = float(os.getenv("AUTO_SOS_SNOOZE_S", "30"))
DEADMAN_RECENT_CRITICAL_S = 10.0
SNAPSHOT_BUCKET = os.getenv("SNAPSHOT_BUCKET", "sos-snapshots")

LEVEL_RANK = {"NORMAL": 0, "MEDIUM": 1, "MODERATE": 1, "CRITICAL": 2}
RANK_LEVEL = {0: "NORMAL", 1: "MODERATE", 2: "CRITICAL"}

SYSTEM_INSTRUCTION = (
    "You are the threat-assessment module of a personal-safety HUD worn by a woman in India. She is the camera.\n"
    "INPUT: image 0 is her full view with ONE person in a red box; then 3-5 consecutive crops of that person "
    "(about 2 s, the last one is the moment that triggered this check); then a sensor note. Judge visible actions, "
    "body language, held objects and vehicles across the whole sequence; use image 0 for context (other people, "
    "vehicles, crowd, indoors). Never judge appearance, clothing, religion, caste, age, gender or skin colour.\n\n"
    "HOW ATTACKS ON WOMEN ARE STAGED (offender research):\n"
    "- BLITZ: sudden violence, no talk: strike, grab, push, choke, tackle, weapon drawn or swung.\n"
    "- SURPRISE: waits or lurks, then closes without speaking, often from the side or behind, out of others' sight.\n"
    "- CON: talks first (time, directions, help, a ride) and gets close; force may follow if she disengages. "
    "A second person may drift in.\n"
    "- VEHICLE: auto, cab, bike or car slows or stops beside her; door opens toward her; occupant beckons or "
    "reaches out; pillion rider leans out or steps off (chain / bag snatching).\n"
    "- STALK/PROBE: keeps reappearing, matches her pace, comes close again after she moved away.\n"
    "- STREET HARASSMENT (common in daylight crowds, buses, metro, markets): leering, walking alongside or just "
    "behind at her pace, remarks or whistles, 'accidental' brushing, groping or pressing in a crowd, pulling her "
    "dupatta or scarf, circling on a two-wheeler, turning hostile after she ignores or refuses. A group may box her in.\n"
    "- AT HOME (most violence against women in India is by a partner or relative): striking, choking, pushing, "
    "dragging, throwing or swinging objects at her is a threat even within family; hugs, closeness or talk are not.\n\n"
    "PRE-ATTACK INDICATORS (each a cue, not proof): repeated glances at her then away; looking around as if "
    "checking for witnesses; a hand hidden behind the back or in clothing while closing in; bladed stance, clenched "
    "fists; sudden change of pace or direction toward her; closing distance while talking; a second person moving "
    "behind or beside her.\n\n"
    "PATTERN labels (set `pattern`):\n"
    "- weapon: knife or blade, OR any held object used or presented as a weapon (chair, stool, bottle, rod, pipe, "
    "stick, brick, bat, tool, helmet or bag swung by its strap, chain, belt, umbrella). Cues: raised above shoulder "
    "or head, cocked back, two-handed grip, swung, thrown or jabbed toward a person. Carried low, at the side, on "
    "the head as a load, or used normally (sitting, drinking, umbrella open in rain) is NOT a weapon.\n"
    "- strike: punching, kicking, slapping, swinging an arm at someone.\n"
    "- grab_snatch: fast reach at her neck, chest, hands, phone, bag, chain or dupatta; tug-of-war over an item; "
    "pillion leaning out. A helmet or mask alone is not a signal.\n"
    "- charge: sprinting or lunging at the camera.\n"
    "- restrain: grabbing arms or clothing, pushing, pinning, choking, dragging, pulling toward a vehicle.\n"
    "- block: standing square in her path, mirroring her moves, arms spread.\n"
    "- concealed: hand hidden behind the back or in clothing while closing in; bladed stance.\n"
    "- con: stranger stops in her space and engages, keeps following if she walks on.\n"
    "- ambush: emerges from the side, behind an object or a parked vehicle and closes fast.\n"
    "- vehicle: see VEHICLE above.\n"
    "- harass: sustained leering while close, pacing her, brushing, groping or pressing against her, or staying "
    "on her after she walked away, with or without words heard.\n"
    "- none: anything else.\n\n"
    "NOT threats: waving, stretching, dancing in a procession or wedding, selfies, friends hugging, a tap on the "
    "shoulder, vendors or beggars approaching with goods or an open palm, auto drivers calling out, pushing while "
    "boarding a crowded bus or train, running for a bus, sports, children playing, people looking both ways to "
    "cross a road, someone asking directions who leaves when she walks on, a tense face alone, ordinary walking.\n\n"
    "BASE RATE: almost everyone near her is busy with their own life. An offender's attention is FIXED on his "
    "target and his path is built around her; a bystander's attention and path are built around his own goal. "
    "Default to 'none'. You are the counter-reasoning step: argue AGAINST a threat first.\n"
    "REASON IN THIS ORDER (fill the fields in this order):\n"
    "1. benign_explanation: the most plausible ordinary explanation (on phone, talking to a companion, carrying "
    "goods, walking to a bus, working, waiting, looking at a shop, crossing).\n"
    "2. attention_on_wearer: true only if his face / eyes point at the camera in most frames. Head bowed, looking "
    "at a phone, at another person, at the road or sideways = false.\n"
    "3. directed_at_wearer: true only if his movement, reach or held object is aimed at HER (closing on her, "
    "reaching toward the camera, blocking her path), not at someone else or nothing.\n"
    "4. severity, only after 1-3.\n\n"
    "SEVERITY:\n"
    "- 'threat': directed_at_wearer AND a hostile act visible in at least one clear frame (object raised, cocked "
    "or swung toward her, strike, grab, restraint, charge, pulled toward a vehicle, blade shown). The benign "
    "explanation must be clearly worse than the hostile one.\n"
    "- 'suspicious': attention_on_wearer across two or more frames PLUS a pre-attack cue (closing while watching her, "
    "hand hidden while approaching, blocking her, pacing her, scanning for witnesses), and the benign explanation "
    "is weak. Violence between OTHER people near her is also 'suspicious' (she should move away).\n"
    "- 'none': ordinary behaviour, or anything the benign explanation covers. Proximity, walking toward her on a "
    "footpath, a loud voice, a tense face, gesturing while talking to someone else, or a dark street are 'none' on "
    "their own.\n"
    "CONFIDENCE = probability your severity is right: 0.8+ only when the action is clear in two or more frames; "
    "0.6 or less when frames are blurry, occluded or the key moment is ambiguous.\n\n"
    "OUTPUT: reason under 10 words describing what is SEEN, not intent. unity_instruction under 8 words, empty unless "
    "'threat': weapon -> 'Weapon raised. Back away. Shout.'; grab_snatch -> 'Secure phone and bag. Step away.'; "
    "strike/charge/restrain -> 'Attack. Shout. Run to people.'; block -> 'Path blocked. Turn back. Call out.'; "
    "con -> 'Do not stop. Walk to people.'; ambush -> 'Move toward people now.'; vehicle -> 'Step back from "
    "vehicle. Move away.'; harass -> 'Keep moving. Go to people.'."
)

_gemini = genai.Client()
_gpu_lock = asyncio.Semaphore(1 if DEVICE == "cuda" else 2)
_active_connections = 0
_features_col = True

YOLO(POSE_WEIGHTS)  # fetch the weights on import
_obj_model = YOLO(OBJ_WEIGHTS) if OBJ_DETECT else None
if _obj_model is not None:
    log.info("Weapon model: %s (classes: %s)", OBJ_WEIGHTS, getattr(_obj_model, "names", "?"))
    try:   # first predict fuses layers / allocates CUDA memory: pay it at boot, not on someone's first frame
        with torch.inference_mode():
            _obj_model.predict(source=np.zeros((480, 640, 3), np.uint8), classes=_ALL_OBJ_IDS, imgsz=OBJ_IMGSZ,
                               conf=OBJ_FLOOR_CONF, device=DEVICE, half=HALF, verbose=False)
    except Exception as _we:
        log.warning("object model warm-up failed: %s", _we)
_obj_lock = asyncio.Lock()


class Severity(str, Enum):
    none = "none"
    suspicious = "suspicious"
    threat = "threat"


class Pattern(str, Enum):
    none = "none"
    weapon = "weapon"
    strike = "strike"
    grab_snatch = "grab_snatch"
    charge = "charge"
    restrain = "restrain"
    block = "block"
    concealed = "concealed"
    con = "con"
    ambush = "ambush"
    vehicle = "vehicle"
    harass = "harass"


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
    pattern: str = ""
    intent: str = ""
    attention: float = -1.0
    distance_m: float
    vx: float = 0.0               # new, optional: box-centre velocity, normalised units / s (client extrapolates)
    vy: float = 0.0
    bbox: BoundaryBox


class DataPayload(BaseModel):
    type: str = "frame"
    timestamp_ms: int
    frame_latency_ms: int
    targets: list[ThreatBox] = []


class GeminiVerdict(BaseModel):
    benign_explanation: str
    attention_on_wearer: bool
    directed_at_wearer: bool
    severity: Severity
    confidence: float = Field(..., ge=0.0, le=1.0)
    pattern: Pattern = Pattern.none
    reason: str
    unity_instruction: str


def _build_gemini_config() -> types.GenerateContentConfig:
    cfg = dict(system_instruction=SYSTEM_INSTRUCTION, response_mime_type="application/json",
               response_schema=GeminiVerdict, temperature=0.1, max_output_tokens=200)
    mr = os.getenv("GEMINI_MEDIA_RES", "").strip().lower()       # low = ~4x fewer image tokens per call
    if mr in ("low", "medium", "high"):
        cfg["media_resolution"] = getattr(types.MediaResolution, f"MEDIA_RESOLUTION_{mr.upper()}")
    tb = os.getenv("GEMINI_THINKING_BUDGET", "").strip()          # 0 = no thinking tokens (cheapest, fastest)
    if tb:
        cfg["thinking_config"] = types.ThinkingConfig(thinking_budget=int(tb))
    return types.GenerateContentConfig(**cfg)


_GEMINI_CONFIG = _build_gemini_config()


# ---------- helpers ----------

_SOF = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
_REDUCED = ((8, cv2.IMREAD_REDUCED_COLOR_8), (4, cv2.IMREAD_REDUCED_COLOR_4), (2, cv2.IMREAD_REDUCED_COLOR_2))


def _jpeg_wh(b: bytes) -> tuple[int, int] | None:
    """(width, height) from the JPEG header without decoding; None if not a JPEG."""
    if b[:2] != b"\xff\xd8":
        return None
    i, n = 2, len(b)
    while i + 9 < n:
        if b[i] != 0xFF:
            i += 1
            continue
        m = b[i + 1]
        if m == 0xFF:
            i += 1
            continue
        if m in (0xD8, 0x01) or 0xD0 <= m <= 0xD7:
            i += 2
            continue
        if m in _SOF:
            return int.from_bytes(b[i + 7:i + 9], "big"), int.from_bytes(b[i + 5:i + 7], "big")
        i += 2 + int.from_bytes(b[i + 2:i + 4], "big")
    return None


def decode_frame(data: bytes) -> np.ndarray | None:
    """Decode, using libjpeg's DCT scaling for big frames (1080p -> 960 costs ~1/3 of a full decode)."""
    flag = cv2.IMREAD_COLOR
    wh = _jpeg_wh(data) if PROC_MAX_SIDE > 0 else None
    if wh:
        side = max(wh)
        for f, fl in _REDUCED:
            if side / f >= PROC_MAX_SIDE:
                flag = fl
                break
    return cv2.imdecode(np.frombuffer(data, np.uint8), flag)


def _gemini_ok(now: float) -> bool:
    while _gemini_calls and now - _gemini_calls[0] > 60.0:
        _gemini_calls.popleft()
    return now >= _gemini_block_until and len(_gemini_calls) < GEMINI_RPM


def _prune(q: deque, now: float, span: float) -> None:
    while q and now - q[0] > span:
        q.popleft()


def _budget_ok(uid: str, now: float) -> bool:
    """Hard caps on EVERY Gemini call, critical included."""
    if not GEMINI_ENABLED:
        return False
    _prune(_gemini_day, now, 86400.0)
    if len(_gemini_day) >= GEMINI_GLOBAL_DAILY:
        return False
    q = _user_gemini.setdefault(uid, deque())
    _prune(q, now, 86400.0)
    if len(q) >= GEMINI_USER_DAILY:
        return False
    return sum(1 for t in q if now - t <= 3600.0) < GEMINI_USER_HOURLY


def _budget_spend(uid: str, now: float) -> None:
    _gemini_day.append(now)
    _user_gemini.setdefault(uid, deque()).append(now)


def _min_conf(kind: str) -> float:
    return GEMINI_SCAN_MIN_CONF if kind == "scan" else GEMINI_MIN_CONF


def _shrink(img: np.ndarray, max_side: int = CROP_MAX_SIDE) -> np.ndarray:
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return img
    s = max_side / m
    return cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)


def _person_crop(frame: np.ndarray, box, wide: bool) -> np.ndarray | None:
    h_img, w_img = frame.shape[:2]
    x1, y1, x2, y2 = box
    px, py = (x2 - x1) * (0.35 if wide else 0.15), (y2 - y1) * (0.30 if wide else 0.10)
    crop = frame[int(max(0, y1 - py)):int(min(h_img, y2 + py)), int(max(0, x1 - px)):int(min(w_img, x2 + px))]
    return _shrink(crop).copy() if crop.size else None


def _thumb_b64(frame: np.ndarray, box) -> str | None:
    crop = _person_crop(frame, box, wide=True)
    if crop is None:
        return None
    ok, buf = cv2.imencode(".jpg", _shrink(crop, SOS_THUMB_SIDE), [cv2.IMWRITE_JPEG_QUALITY, 70])
    return base64.b64encode(buf.tobytes()).decode() if ok else None


def _context_image(frame: np.ndarray, box) -> np.ndarray:
    img = _shrink(frame, CONTEXT_MAX_SIDE).copy()
    s = img.shape[1] / frame.shape[1]
    bx = tuple(int(v * s) for v in box)
    img = blur_bystanders(img, bx)
    cv2.rectangle(img, bx[:2], bx[2:], (0, 0, 255), max(2, img.shape[1] // 200))
    return img


def _embed_many(frame: np.ndarray, boxes: list) -> list:
    out = []
    for x1, y1, x2, y2 in boxes:
        try:
            out.append(embed_person(frame, x1, y1, x2, y2))
        except Exception:
            out.append(None)
    return out


def _jsonable(obj):
    return json.loads(json.dumps(obj, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


def _new_track(t: float, hfov: float) -> dict:
    return {"first": t, "intent": IntentTracker(hfov), "last_crop": 0.0, "pose": deque(), "whits": {}, "dist_f": Median3(),
            "gaze": GazeTracker(), "episodes": ApproachEpisodes(), "along": deque(), "near_since": None,
            "ranks": deque(maxlen=3), "hold": (0, 0.0, "", 0.0), "rider": ("", False), "rider_until": 0.0,
            "last_dbg": 0.0, "crit_hit": 0.0, "sos_dbg": 0.0,
            "ident": None, "last_reid": -1e9, "vel": None}


def _stabilise(st: dict, rank: int, score: float, reason: str, t: float, clear: bool = False):
    st["ranks"].append(rank)
    if clear:
        st["hold"] = (0, 0.0, "", 0.0)
    if rank == 1 and sum(1 for r in st["ranks"] if r >= 1) < 2:
        rank, score, reason = 0, min(score, 0.3), ""
    h_rank, h_until, h_reason, h_score = st["hold"]
    if h_rank > rank and t < h_until:
        return h_rank, max(score, h_score), h_reason
    if rank >= 1:
        st["hold"] = (rank, t + (HOLD_CRITICAL_S if rank >= 2 else HOLD_MODERATE_S), reason, score)
    return rank, score, reason


def _velocity(st: dict, t: float, cx: float, cy: float) -> tuple[float, float]:
    pv = st["vel"]
    vx = vy = 0.0
    if pv is not None and 0.0 < t - pv[0] <= 0.5:
        dt = t - pv[0]
        vx = VEL_EMA * (cx - pv[1]) / dt + (1 - VEL_EMA) * pv[3]
        vy = VEL_EMA * (cy - pv[2]) / dt + (1 - VEL_EMA) * pv[4]
        vx, vy = max(-VEL_MAX, min(VEL_MAX, vx)), max(-VEL_MAX, min(VEL_MAX, vy))
    st["vel"] = (t, cx, cy, vx, vy)
    return round(vx, 4), round(vy, 4)


def detect_objects(frame: np.ndarray, near_boxes: list) -> list:
    """Full-frame pass (hand-held objects + vehicles) plus an upscaled pass on the closest people's crops."""
    with torch.inference_mode():
        res = _obj_model.predict(source=frame, classes=_ALL_OBJ_IDS, imgsz=OBJ_IMGSZ, conf=OBJ_FLOOR_CONF,
                                 device=DEVICE, half=HALF, verbose=False)
        found = parse_objects(res[0]) if res else []
        if OBJ_CROP_PASS and near_boxes:
            h, w = frame.shape[:2]
            crops, offs = [], []
            for x1, y1, x2, y2 in near_boxes[:OBJ_CROP_MAX_PEOPLE]:
                bw, bh = x2 - x1, y2 - y1
                cx1, cy1 = int(max(0, x1 - 0.35 * bw)), int(max(0, y1 - 0.3 * bh))
                cx2, cy2 = int(min(w, x2 + 0.35 * bw)), int(min(h, y2 + 0.1 * bh))
                if cx2 - cx1 >= 16 and cy2 - cy1 >= 16:
                    crops.append(np.ascontiguousarray(frame[cy1:cy2, cx1:cx2]))
                    offs.append((cx1, cy1))
            if crops:
                rs = _obj_model.predict(source=crops, classes=_HAND_OBJ_IDS, imgsz=OBJ_CROP_IMGSZ,
                                        conf=OBJ_FLOOR_CONF, device=DEVICE, half=HALF, verbose=False)
                for r, (dx, dy) in zip(rs, offs):
                    found += parse_objects(r, dx, dy)
    return merge_objects(found)


def _encode_contents(crops: list, dist_m: float, hint: str, ctx: str, scene: np.ndarray | None) -> list | None:
    """Runs in a worker thread: JPEG-encode everything Gemini sees."""
    contents: list = []
    if scene is not None:
        ok, buf = cv2.imencode(".jpg", scene, [cv2.IMWRITE_JPEG_QUALITY, CONTEXT_JPEG_Q])
        if ok:
            contents += ["Image 0: her full view. The person to assess is in the red box.",
                         types.Part.from_bytes(data=buf.tobytes(), mime_type="image/jpeg")]
    for index, crop in enumerate(crops, start=1):
        ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, CROP_JPEG_Q])
        if not ok:
            return None
        tag = " (trigger moment)" if index == len(crops) else ""
        contents += [f"Sequence frame {index} of {len(crops)}{tag}.",
                     types.Part.from_bytes(data=buf.tobytes(), mime_type="image/jpeg")]
    contents.append(
        f"The person is about {dist_m:.1f} m from the wearer. Assess the visible action across the whole "
        "sequence, not one isolated pose. Do not infer intent from proximity alone. Keep reason under 10 words "
        "and unity_instruction under 8 words; unity_instruction must be empty unless severity is 'threat'.")
    if ctx:
        contents.append(f"Sensor note (context only, not evidence of intent): {ctx}.")
    if hint:
        contents.append(f"An object detector (often wrong: phones and pens read as knives) flagged a possible "
                        f"{hint} near this person's hand. First check it is really there. Judge HOW it is held: "
                        "raised above the shoulder, cocked back, swung, thrown or jabbed toward someone = 'threat'; "
                        "carried low, at the side, as a load or used normally = not a threat.")
    return contents


async def verify_with_gemini(crops: list, dist_m: float, hint: str = "", ctx: str = "",
                             scene: np.ndarray | None = None) -> tuple[str, float, str, str, str] | None:
    """-> (severity, confidence, reason, instruction, pattern), or None if the check itself failed.
    None = 'no information', never 'safe': callers must not downgrade on it."""
    global _gemini_block_until
    if len(crops) < TEMPORAL_MIN_FRAMES:
        return None
    try:
        contents = await asyncio.to_thread(_encode_contents, crops, dist_m, hint, ctx, scene)
        if contents is None:
            return None
        async with _gemini_global:
            resp = await asyncio.wait_for(
                _gemini.aio.models.generate_content(model=GEMINI_MODEL, contents=contents, config=_GEMINI_CONFIG),
                timeout=5.0)
        verdict: GeminiVerdict | None = resp.parsed
        if verdict is None:
            verdict = GeminiVerdict.model_validate_json(resp.text)
        sev = verdict.severity.value
        if sev == "threat" and not verdict.directed_at_wearer:
            sev = "suspicious"
        if (sev == "suspicious" and not verdict.attention_on_wearer and not verdict.directed_at_wearer
                and verdict.pattern not in (Pattern.weapon, Pattern.strike, Pattern.restrain)):
            sev = "none"
        log.info("gemini sev=%s (raw %s) pat=%s conf=%.2f attn=%s dir=%s | %s | benign: %s", sev,
                 verdict.severity.value, verdict.pattern.value, verdict.confidence, verdict.attention_on_wearer,
                 verdict.directed_at_wearer, verdict.reason, verdict.benign_explanation[:80])
        return (sev, float(verdict.confidence), verdict.reason[:80],
                verdict.unity_instruction[:60] if sev == "threat" else "", verdict.pattern.value)
    except asyncio.TimeoutError:
        log.info("gemini verify timeout")
        return None
    except Exception as e:
        msg = str(e)
        if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
            m = re.search(r"retry in ([\d.]+)s", msg)
            delay = float(m.group(1)) if m else 30.0
            _gemini_block_until = time.time() + delay + 1.0
            log.warning("gemini quota hit; pausing checks for %.0fs", delay)
        else:
            log.warning("gemini verify error: %s", e)
        return None


async def gemini_check(crops: list, dist_m: float, hint: str, ctx: str, frame: np.ndarray, box):
    """Builds the boxed + bystander-blurred context image off the event loop, then verifies."""
    try:
        scene_img = await asyncio.to_thread(_context_image, frame, box)
    except Exception as e:
        log.debug("context image failed: %s", e)
        scene_img = None
    return await verify_with_gemini(crops, dist_m, hint, ctx, scene_img)


async def upload_snapshot(user_id: str, frame: np.ndarray) -> str | None:
    try:
        ok, buf = await asyncio.to_thread(cv2.imencode, ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            return None
        path = f"{user_id}/{int(time.time() * 1000)}.jpg"
        await db(lambda: supabase_admin.storage.from_(SNAPSHOT_BUCKET).upload(
            path, buf.tobytes(), {"content-type": "image/jpeg"}))
        return f"{SNAPSHOT_BUCKET}/{path}"
    except Exception as e:
        log.error("snapshot upload failed: %s", e)
        return None


async def log_threat_event(user_id: str, track_id: str, level: str, source: str, score: float,
                           reason: str, distance_m: float | None, approach_mps: float | None,
                           verified: bool, gps: dict, sink: deque | None = None,
                           features: dict | None = None) -> None:
    """Best-effort insert into public.threat_events. Never raises."""
    global _features_col
    row = {
        "user_id": user_id, "track_id": track_id, "level": level, "source": source, "score": float(score),
        "reason": (reason or "")[:200],
        "distance_m": None if distance_m is None else float(distance_m),
        "approach_mps": None if approach_mps is None else float(approach_mps),
        "verified_by_gemini": bool(verified),
        "latitude": gps.get("latitude"), "longitude": gps.get("longitude"),
    }
    if features is not None and _features_col:
        row["features"] = _jsonable(features)
    try:
        try:
            res = await db(lambda: supabase_admin.table("threat_events").insert(row).execute())
        except Exception as e:
            if "features" not in row or "features" not in str(e):
                raise
            _features_col = False
            log.warning("threat_events.features missing - run migrations/2026_10_threat_features.sql")
            row.pop("features", None)
            res = await db(lambda: supabase_admin.table("threat_events").insert(row).execute())
        data = getattr(res, "data", None)
        if sink is not None and data:
            sink.append((time.time(), data[0]["id"], track_id))
    except Exception as e:
        log.error("threat_events insert failed: %s", e)


async def label_threat_events(user_id: str, ids: list, label: str) -> None:
    try:
        await db(lambda: supabase_admin.table("threat_events")
                 .update({"user_label": label, "labeled_at": datetime.now(timezone.utc).isoformat()})
                 .in_("id", ids).eq("user_id", user_id).execute())
    except Exception as e:
        log.error("threat_events label failed: %s", e)


def _load_pose_model() -> YOLO:
    """Per-connection model (ByteTrack state is per model). Loaded + warmed in a thread."""
    m = YOLO(POSE_WEIGHTS)
    with torch.inference_mode():
        m.track(source=np.zeros((360, 640, 3), np.uint8), persist=True, tracker="bytetrack.yaml", classes=[0],
                imgsz=YOLO_IMGSZ, device=DEVICE, half=HALF, verbose=False, conf=YOLO_CONF)
    return m


@router.websocket("/prod/main")
async def main_suraksha(websocket: WebSocket, token: str = Query(None),
                        hfov: float = Query(65.0, ge=20.0, le=150.0)):
    global _active_connections

    auth_hdr = websocket.headers.get("authorization", "")
    if auth_hdr.lower().startswith("bearer "):
        token = auth_hdr[7:].strip()
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
    try:
        async with _gpu_lock:
            model = await asyncio.to_thread(_load_pose_model)   # was blocking EVERY connection's event loop
    except Exception as e:
        log.exception("pose model load failed: %s", e)
        _active_connections -= 1
        await websocket.close(code=status.WS_1011_INTERNAL_ERROR, reason="Model load failed")
        return

    cooldown: dict[str, float] = {}
    cleared: dict[str, tuple[float, frozenset]] = {}
    my_calls: deque = deque()
    last_scan = 0.0
    pending: dict[str, tuple[asyncio.Task, str]] = {}
    verdicts: dict[str, tuple] = {}
    crop_history: dict[str, deque] = {}
    tstate: dict[str, dict] = {}
    engine = ThreatEngine(hfov=hfov)
    gallery = PersonGallery()
    follow_test = FollowTest()
    crowd = CrowdBaseline()
    clock = FrameClock()
    latest: tuple[bytes, float, float | None] | None = None
    new_frame = asyncio.Event()
    closed = False
    send_lock = asyncio.Lock()
    company = False
    last_frame: np.ndarray | None = None

    # pipeline slots (newest wins, stale work is dropped, never queued)
    infer_slot: tuple | None = None          # (frame, dets_raw, ft, recv_ts)
    infer_ready = asyncio.Event()
    obj_req: tuple | None = None             # (frame, ft, near_boxes)
    obj_wake = asyncio.Event()
    obj_cache: tuple[float, list] = (0.0, [])
    audio_busy = False

    # auto-SOS state
    gps: dict = {}
    scene = SceneContext()
    scene_ctx: dict = {"dark": False, "safe": False, "brightness": 0.0}
    scene_ts = -1e9
    confirmations: dict[str, list[float]] = {}
    countdown_task: asyncio.Task | None = None
    armed_frame: np.ndarray | None = None
    armed_reason = ""
    armed_details: dict = {}
    armed_track = ""
    snoozed: dict[str, float] = {}
    fired_incident: dict | None = None
    firing: asyncio.Task | None = None
    last_fire_ts = 0.0
    last_fire_ok = False
    scream_val, scream_ts = 0.0, 0.0
    audio_hits: list[float] = []
    audio_alert_sent = 0.0
    logged: dict[str, tuple[float, int, bool]] = {}
    last_seen: dict[str, float] = {}
    recent_events: deque = deque(maxlen=50)
    last_critical_ts = 0.0
    last_critical_reason = ""
    last_budget_log = 0.0
    last_manual_ts = 0.0
    session_start = time.time()

    async def send_json(obj: dict | BaseModel):
        text = obj.model_dump_json() if isinstance(obj, BaseModel) else json.dumps(obj)
        async with send_lock:
            await websocket.send_text(text)

    async def try_send(obj: dict) -> None:
        try:
            await send_json(obj)
        except Exception as e:
            log.info("send failed (socket closing?): %s", e)

    def gps_details() -> dict:
        return {"latitude": gps.get("latitude"), "longitude": gps.get("longitude"),
                "battery_percentage": gps.get("battery_percentage")}

    def fire_busy() -> bool:
        return firing is not None and not firing.done()

    def recently_fired(now: float) -> bool:
        return last_fire_ok and now - last_fire_ts < AUTO_SOS_REARM_S

    def can_prompt(now: float, track_id: str | None = None) -> bool:
        if not AUTO_SOS_ENABLED or countdown_task is not None or fire_busy() or recently_fired(now):
            return False
        if track_id is not None and now < snoozed.get(track_id, 0.0):
            return False
        return True

    async def do_fire(frame_img: np.ndarray | None, reason: str, source: str = "AUTO_HUD") -> dict | None:
        nonlocal fired_incident, last_fire_ts, last_fire_ok
        snap = await upload_snapshot(user.id, frame_img) if frame_img is not None else None
        try:
            inc = await fire_sos(
                user,
                latitude=gps.get("latitude", float(DEMO_LAT) if DEMO_LAT else None),
                longitude=gps.get("longitude", float(DEMO_LNG) if DEMO_LNG else None),
                accuracy_m=gps.get("accuracy_m", 0.0), speed_mps=gps.get("speed_mps", 0.0),
                battery_percentage=gps.get("battery_percentage"),
                state_code=gps.get("state_code", "NATIONAL"), district=gps.get("district", "All"),
                snapshot_url=snap, threat_level="CRITICAL", source=source,
            )
            fired_incident = inc
            last_fire_ts, last_fire_ok = time.time(), True
            confirmations.clear()
            log.warning("AUTO SOS user=%s reason=%s", user.id, reason)
            return inc
        except SOSFireError as e:
            last_fire_ts, last_fire_ok = time.time(), False
            log.error("auto SOS failed: %s", e.detail)
            return None
        except Exception as e:
            last_fire_ts, last_fire_ok = time.time(), False
            log.error("auto SOS crashed: %s", e)
            return None

    def start_fire(reason: str, frame_img: np.ndarray | None = None, source: str = "AUTO_HUD") -> asyncio.Task:
        nonlocal firing
        if not fire_busy():
            firing = spawn(do_fire(frame_img if frame_img is not None else armed_frame, reason, source))
        return firing

    async def fire_and_report(reason: str, frame_img: np.ndarray | None = None, source: str = "AUTO_HUD"):
        try:
            inc = await asyncio.shield(start_fire(reason, frame_img, source))
        except Exception as e:
            log.error("SOS fire task failed: %s", e)
            inc = None
        await try_send({"type": "sos_fired", "incident": inc,
                        "unity_instruction": "SOS sent. Help is being alerted."} if inc else
                       {"type": "sos_failed", "client_should_dial": "112",
                        "unity_instruction": "SOS failed. Dial 112 now."})

    async def countdown():
        nonlocal countdown_task
        me = asyncio.current_task()
        try:
            await try_send({"type": "sos_prompt", "seconds": AUTO_SOS_COUNTDOWN_S,
                            "title": "Potential threat detected.", "reason": armed_reason,
                            "details": armed_details,
                            "unity_instruction": "Possible threat. Cancel if you are safe."})
            await asyncio.sleep(AUTO_SOS_COUNTDOWN_S + AUTO_SOS_GRACE_S)
            if countdown_task is me:
                countdown_task = None
            await fire_and_report(armed_reason + " (app did not answer)")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.warning("countdown interrupted: %s", e)
        finally:
            if countdown_task is me:
                countdown_task = None

    def maybe_arm(track_id: str, now: float, frame_img: np.ndarray, box, reason: str, details: dict):
        nonlocal countdown_task, armed_frame, armed_reason, armed_details, armed_track
        if not can_prompt(now, track_id):
            return
        hits = confirmations.setdefault(track_id, [])
        hits[:] = [s for s in hits if now - s <= AUTO_SOS_WINDOW_S]
        hits.append(now)
        screaming = scream_val >= AUTO_SOS_SCREAM_MIN and now - scream_ts <= AUTO_SOS_SCREAM_FRESH_S
        needed = AUTO_SOS_SCREAM_CONFIRMATIONS if screaming else AUTO_SOS_CONFIRMATIONS
        if len(hits) >= needed:
            n_hits = len(hits)
            confirmations.pop(track_id, None)
            armed_track = track_id
            armed_frame = frame_img                  # frames are never written to: no copy needed
            armed_reason = f"Critical behavior ({track_id}){' + scream' if screaming else ''}: {reason}"
            armed_details = {**details, "source": "video", "confirmations": n_hits,
                             "scream": round(scream_val, 2) if screaming else 0.0,
                             "thumb_b64": _thumb_b64(frame_img, box), **gps_details()}
            log.warning("SOS POPUP user=%s track=%s %s", user.id, track_id,
                        {k: v for k, v in details.items() if k != "thumb_b64"})
            countdown_task = asyncio.create_task(countdown())

    def apply_scream(val: float) -> None:
        nonlocal scream_val, scream_ts
        scream_val = val
        scream_ts = time.time()
        engine.set_audio(scream_val)
        if scream_val >= AUDIO_ALERT_MIN:
            audio_hits.append(scream_ts)
            audio_hits[:] = [s for s in audio_hits if scream_ts - s <= AUDIO_ALERT_WINDOW_S]

    async def classify_audio(audio_np: np.ndarray, sr: int) -> None:
        """PANNs in a worker thread (was run inline: froze frame intake for ~100 ms per chunk)."""
        nonlocal audio_busy
        try:
            events = await asyncio.to_thread(get_audio_classifier().classify, audio_np, sr=sr)
            for evt_label, evt_conf in events:
                if evt_label == "scream" and evt_conf >= 0.5:
                    apply_scream(max(scream_val, evt_conf))
                elif evt_label == "gunshot" and evt_conf >= 0.55:
                    engine.set_audio(evt_conf)
        except Exception as _ae:
            log.debug("audio_raw classify error: %s", _ae)
        finally:
            audio_busy = False

    async def trigger_manual() -> None:
        """Manual SOS: debounced, and never uploads a snapshot or hits the DB for a repeat press."""
        nonlocal last_manual_ts, countdown_task
        now = time.time()
        if now - last_manual_ts < MANUAL_MIN_GAP_S or fire_busy() or recently_fired(now):
            await send_json({"type": "sos_already_sent", "incident": fired_incident})
            return
        last_manual_ts = now
        if countdown_task and not countdown_task.done():
            countdown_task.cancel()
            countdown_task = None
        spawn(fire_and_report("Manual SOS", last_frame, "MANUAL"))

    async def handle_control(msg: dict):
        nonlocal countdown_task, fired_incident, company, audio_busy
        nonlocal armed_frame, armed_reason, armed_details, armed_track
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
            heading = msg.get("heading_deg")
            follow_test.user(time.time(), float(gps.get("speed_mps", 0.0) or 0.0),
                             float(heading) % 360.0 if isinstance(heading, (int, float)) else None)
        elif kind == "safe_zones":
            scene.set_zones(msg.get("zones"))
        elif kind == "safe_mode":
            scene.manual_safe = bool(msg.get("on"))
        elif kind == "company_mode":
            company = bool(msg.get("on"))
        elif kind == "audio":
            if isinstance(msg.get("scream"), (int, float)):
                apply_scream(max(0.0, min(1.0, float(msg["scream"]))))
        elif kind == "audio_raw":
            pcm_b64 = msg.get("pcm_b64")
            if isinstance(pcm_b64, str) and not audio_busy:      # one chunk in flight; newer chunks win next
                try:
                    audio_np = np.frombuffer(base64.b64decode(pcm_b64), dtype=np.int16)
                    audio_busy = True
                    spawn(classify_audio(audio_np, int(msg.get("sr", 16000))))
                except Exception as _ae:
                    audio_busy = False
                    log.debug("audio_raw decode error: %s", _ae)
        elif kind == "speech":
            txt = msg.get("text")
            if isinstance(txt, str) and txt:
                lvl, words = classify_speech(txt[:300])
                if lvl:
                    engine.set_verbal(lvl, words)
        elif kind == "confirm_sos":
            if countdown_task and not countdown_task.done() and not fire_busy():
                countdown_task.cancel()
                countdown_task = None
                log.warning("SOS confirmed by user %s", user.id)
                spawn(fire_and_report(armed_reason + " (confirmed by user)"))
            elif fire_busy() or recently_fired(time.time()):
                await send_json({"type": "sos_already_sent", "incident": fired_incident})
            else:
                log.warning("SOS confirmed with no server popup (manual) user %s", user.id)
                await trigger_manual()
        elif kind == "cancel_sos":
            if countdown_task and not countdown_task.done() and not fire_busy():
                countdown_task.cancel()
                countdown_task = None
                confirmations.pop(armed_track, None)
                snoozed[armed_track] = time.time() + AUTO_SOS_SNOOZE_S
                await send_json({"type": "sos_cancelled", "snooze_s": AUTO_SOS_SNOOZE_S,
                                 "unity_instruction": "SOS cancelled."})
            elif fire_busy() or recently_fired(time.time()):
                await send_json({"type": "sos_already_sent", "incident": fired_incident})
            else:
                await send_json({"type": "sos_cancelled", "snooze_s": 0, "unity_instruction": ""})
        elif kind == "feedback":
            label = msg.get("label")
            if label not in FEEDBACK_LABELS:
                return
            if label == "false_alarm" and countdown_task and not countdown_task.done() and not fire_busy():
                await handle_control({"type": "cancel_sos"})
            cutoff = time.time() - FEEDBACK_WINDOW_S
            tid = msg.get("track_id") if isinstance(msg.get("track_id"), str) else None
            ids = [eid for ts, eid, trk in recent_events if ts >= cutoff and (tid is None or trk == tid)]
            if ids:
                spawn(label_threat_events(user.id, ids, label))
            await send_json({"type": "feedback_ack", "label": label, "events_labeled": len(ids)})
        elif kind == "test_popup" and os.getenv("SOS_TEST_POPUP", "0") == "1":
            if countdown_task is None and not fire_busy():
                armed_track = "test"
                armed_frame = last_frame
                armed_reason = "TEST popup (no real threat)"
                armed_details = {"source": "test", "pattern": "weapon", "confidence": 0.95,
                                 "instruction": "Test only.", "thumb_b64": None, **gps_details()}
                countdown_task = asyncio.create_task(countdown())
        elif kind == "manual_sos":
            await trigger_manual()

    def infer(frame: np.ndarray):
        """Pose + ByteTrack, and the GPU->CPU copies, all inside the worker thread."""
        with torch.inference_mode():
            r = model.track(source=frame, persist=True, tracker="bytetrack.yaml", classes=[0],
                            imgsz=YOLO_IMGSZ, device=DEVICE, half=HALF, verbose=False, conf=YOLO_CONF)
        r0 = r[0] if r else None
        if r0 is None or r0.boxes is None or len(r0.boxes) == 0 or r0.boxes.id is None:
            return None
        xyxy = r0.boxes.xyxy.cpu().numpy()
        ids = r0.boxes.id.cpu().numpy().astype(int)
        kxy = kcf = None
        if r0.keypoints is not None:
            kxy = r0.keypoints.xy.cpu().numpy()
            if r0.keypoints.conf is not None:
                kcf = r0.keypoints.conf.cpu().numpy()
        return xyxy, ids, kxy, kcf

    def decode_and_infer(jpeg: bytes):
        frame = decode_frame(jpeg)
        return (frame, None) if frame is None else (frame, infer(frame))

    ctl_times: deque = deque()

    async def receiver():
        nonlocal latest, closed
        try:
            while True:
                msg = await websocket.receive()
                if msg["type"] == "websocket.disconnect":
                    break
                if msg.get("bytes") is not None:
                    data = msg["bytes"]
                    if len(data) <= MAX_FRAME_BYTES:
                        cap = None
                        if data[:4] == FRAME_MAGIC and len(data) > 12:
                            cap = int.from_bytes(data[4:12], "little") / 1000.0
                            data = data[12:]
                        latest = (data, time.time(), cap)
                        new_frame.set()
                elif msg.get("text"):
                    try:
                        m = json.loads(msg["text"])
                    except (ValueError, TypeError):
                        continue
                    if not isinstance(m, dict):
                        continue
                    if m.get("type") not in ("manual_sos", "confirm_sos", "cancel_sos"):   # never throttle SOS
                        tnow = time.monotonic()
                        while ctl_times and tnow - ctl_times[0] > 1.0:
                            ctl_times.popleft()
                        if len(ctl_times) >= CONTROL_MSGS_PER_S:
                            continue
                        ctl_times.append(tnow)
                    await handle_control(m)
        except Exception:
            pass
        finally:
            closed = True
            new_frame.set()
            obj_wake.set()
            infer_ready.set()

    async def infer_loop():
        """Stage 1: newest JPEG -> decode + pose/track (one thread hop). Runs while stage 2 processes."""
        nonlocal latest, infer_slot
        try:
            while not closed:
                await new_frame.wait()
                new_frame.clear()
                if closed:
                    break
                if latest is None:
                    continue
                jpeg, recv_ts, cap_ts = latest
                latest = None
                async with _gpu_lock:
                    frame, raw = await asyncio.to_thread(decode_and_infer, jpeg)
                if frame is None:
                    continue
                ft = clock.stamp(recv_ts, cap_ts)
                infer_slot = (frame, raw, ft, recv_ts)
                infer_ready.set()
        except Exception as e:
            log.exception("infer loop failed (%s): %s", user.id, e)
        finally:
            infer_ready.set()

    async def obj_loop():
        """Objects / vehicles on the newest frame, time-gated, never blocking the HUD frame loop."""
        nonlocal obj_req, obj_cache
        last_run = -1e9
        while not closed:
            await obj_wake.wait()
            obj_wake.clear()
            if closed:
                break
            wait = OBJ_MIN_GAP_S - (time.monotonic() - last_run)
            if wait > 0:
                await asyncio.sleep(wait)
            req, obj_req = obj_req, None
            if req is None:
                continue
            f, f_ts, near = req
            last_run = time.monotonic()
            try:
                async with _gpu_lock:
                    async with _obj_lock:
                        found = await asyncio.to_thread(detect_objects, f, near)
                obj_cache = (f_ts, found)
            except Exception as e:
                log.warning("object pass failed: %s", e)

    recv_task = asyncio.create_task(receiver())
    infer_task = asyncio.create_task(infer_loop())
    obj_task = asyncio.create_task(obj_loop()) if _obj_model is not None else None

    try:
        while True:
            await infer_ready.wait()
            infer_ready.clear()
            if infer_slot is None:
                if closed or infer_task.done():
                    break
                continue
            frame, raw, ft, recv_ts = infer_slot
            infer_slot = None

            if (MAX_SESSION_S and time.time() - session_start > MAX_SESSION_S
                    and countdown_task is None and not fire_busy()):
                last_critical_ts = 0.0          # a planned close must not trigger the dead-man switch
                await try_send({"type": "session_limit", "unity_instruction": "Session limit reached. Reconnect."})
                break

            h_img, w_img = frame.shape[:2]
            last_frame = frame
            if ft - scene_ts >= SCENE_EVERY_S:
                scene_ctx = scene.update(frame, gps)
                scene_ts = ft
            user_speed = float(gps.get("speed_mps", 0.0) or 0.0)

            # ---- A. per-person features ----
            dets: list[dict] = []
            if raw is not None:
                xyxy, ids, kxy_all, kcf_all = raw
                for i in range(min(len(xyxy), len(ids))):
                    key = f"vector_{int(ids[i]):02d}"
                    x1, y1, x2, y2 = (float(v) for v in xyxy[i])
                    x1, y1, x2, y2 = max(0.0, x1), max(0.0, y1), min(float(w_img), x2), min(float(h_img), y2)
                    bw, bh = x2 - x1, y2 - y1
                    if bw <= 1 or bh <= 1:
                        continue
                    kxy = kxy_all[i] if kxy_all is not None and i < len(kxy_all) else None
                    kcf = kcf_all[i] if kcf_all is not None and i < len(kcf_all) else None
                    st = tstate.setdefault(key, _new_track(ft, hfov))
                    dist = round(st["dist_f"](estimate_distance(kxy, kcf, bh, w_img, h_img, hfov, box_w_px=bw)), 2)
                    p_score, p_why = pose_signals(st["pose"], kxy, kcf, ft, bh)
                    st["gaze"].update(ft, head_state(kxy, kcf, bh))
                    vx, vy = _velocity(st, ft, (x1 + bw / 2) / w_img, (y1 + bh / 2) / h_img)
                    dets.append({"key": key, "box": (x1, y1, x2, y2), "bw": bw, "bh": bh, "kxy": kxy, "kcf": kcf,
                                 "dist": dist, "p": p_score, "p_why": p_why, "st": st, "vx": vx, "vy": vy,
                                 "reid_ok": bh >= REID_MIN_BOX_H and dist <= FOLLOW_MAX_DIST_M * 2,
                                 "pos": locate((x1 + bw / 2) / w_img, dist, hfov)})

            # ---- B. hand the newest frame to the object task; batch re-ID in one thread hop ----
            if obj_task is not None and dets:
                obj_req = (frame, ft, [d["box"] for d in sorted(dets, key=lambda d: d["dist"])
                                       if d["dist"] <= OBJ_CROP_MAX_DIST_M])
                obj_wake.set()
            need = [d for d in dets if d["reid_ok"] and ft - d["st"]["last_reid"] >= REID_EVERY_S]
            if need:
                embs = await asyncio.to_thread(_embed_many, frame, [d["box"] for d in need])
                for d, e in zip(need, embs):
                    d["st"]["last_reid"] = ft
                    d["emb"] = e

            obj_age = ft - obj_cache[0]
            objs = obj_cache[1] if 0.0 <= obj_age <= OBJ_CACHE_TTL_S else []
            moving = [o for o in objs if o[0] in _MOVING] if obj_age <= VEHICLE_FRESH_S else []
            flags = ride_flags([d["box"] for d in dets], moving) if moving else None
            for j, d in enumerate(dets):
                if flags and flags[j][0]:
                    d["st"]["rider"], d["st"]["rider_until"] = flags[j], ft + 2.0
                elif ft > d["st"]["rider_until"]:
                    d["st"]["rider"] = ("", False)
            crowd_active = crowd.update(ft, [d["p"] for d in dets])

            # ---- C. per-track assessment ----
            wall = time.time()
            targets: list[ThreatBox] = []
            seen: set[str] = set()
            frame_keys = {d["key"] for d in dets}
            candidates: list[tuple] = []

            for d in dets:
                key, st, dist = d["key"], d["st"], d["dist"]
                x1, y1, x2, y2 = d["box"]
                bw, bh, kxy, kcf = d["bw"], d["bh"], d["kxy"], d["kcf"]
                seen.add(key)
                cx_norm = (x1 + bw / 2) / w_img
                p_score, p_why = d["p"], d["p_why"]
                if crowd_active and dist > CROWD_DISCOUNT_MIN_DIST_M and p_score > 0.25:
                    p_score, p_why = 0.25, ""

                # 0. objects: held != threat. Sharp in hand, raised, or swung is.
                obj = match_object(objs, d["box"], kxy, kcf) if objs else None
                if obj is not None:
                    hits = st["whits"].setdefault(obj["label"], [])
                    if not hits or hits[-1] != obj_cache[0]:
                        hits.append(obj_cache[0])
                for lbl in list(st["whits"]):
                    st["whits"][lbl] = [h for h in st["whits"][lbl] if ft - h <= WEAPON_HIT_WINDOW_S]
                    if not st["whits"][lbl]:
                        del st["whits"][lbl]
                obj_label, held, raised = "", False, False
                extra: dict[str, float] = {}
                if obj is not None and (obj["held"] or obj["label"] in SHARP):
                    need_hits = 1 if (p_score >= 0.6 or obj["raised"]) else WEAPON_MIN_HITS
                    if len(st["whits"].get(obj["label"], [])) >= need_hits:
                        obj_label, held, raised = obj["label"], obj["held"], obj["raised"]
                        if held and obj_label in SHARP:
                            extra["weapon_held"] = 1.0
                        elif raised:
                            extra["object_held"] = 1.0
                obj_alert = obj_label in SHARP or raised

                # 1. behaviour
                assess = engine.update(key, ft, dist, cx_norm, bw, bh, kxy, kcf, KP_CONF, ext_pose=(p_score, p_why))
                actx = assess["ctx"]
                approach = float(assess["approach_mps"])
                isolated = bool(actx.get("isolated"))
                cats = dict(assess["cats"])
                cats.pop("follow", None)
                cats.update(extra)
                if held and (p_score >= 0.6 or cats.keys() & {"strike", "contact"} or (raised and p_score >= 0.3)):
                    cats["object_swing"] = 1.0
                if company:
                    for k in SOCIAL_CATS:
                        cats.pop(k, None)
                    if p_score < 0.5:
                        cats.pop("contact", None)

                # following: re-ID (cached between refreshes), TEDD stop/turn, pacing alongside / looking back
                if not d["reid_ok"]:
                    ident = None
                elif "emb" in d:
                    ident = gallery.assign(key, ft, d["emb"], dist, frame_keys) if d["emb"] is not None else None
                    st["ident"] = ident
                else:
                    ident = st["ident"]
                follow_lvl = 0
                if ident is not None:
                    f_rank, f_score, _f_why = gallery.persistence(ident, ft, user_speed)
                    if f_rank > 0:
                        follow_lvl = 2 if f_score >= 0.65 else 1
                follow_test.track(key, ft, dist, approach)
                follow_lvl = max(follow_lvl, min(2, follow_test.level(key)))
                along = st["along"]
                along.append((ft, dist <= ALONGSIDE_MAX_DIST_M
                               and (cx_norm <= ALONGSIDE_EDGE or cx_norm >= 1 - ALONGSIDE_EDGE)))
                while along and ft - along[0][0] > FOLLOW_SECONDS:
                    along.popleft()
                glances, scans = st["gaze"].glances(ft), st["gaze"].scans(ft)
                if (follow_lvl == 0 and user_speed >= FOLLOW_MIN_USER_SPEED and dist <= FOLLOW_MAX_DIST_M
                        and ft - st["first"] >= FOLLOW_SECONDS
                        and (sum(1 for _, a in along if a) >= ALONGSIDE_FRAC * len(along) or glances >= 2)):
                    follow_lvl = 1
                if follow_lvl and not company:
                    cats["follow_strong" if follow_lvl >= 2 else "follow"] = 1.0

                # COUNTER-REASONING
                episodes = st["episodes"].update(ft, dist)
                it = st["intent"].update(ft, kxy, kcf, KP_CONF, bh, cx_norm, dist,
                                         others=[o["pos"] for o in dets if o is not d],
                                         follow_lvl=follow_lvl, reapproach=episodes)
                istate = it["state"]
                directed = istate == "targeted" or it["aimed"] or dist <= DIRECT_DIST_M
                if istate != "targeted":
                    for k in SOCIAL_GATED:
                        if not (k == "stare" and istate == "watching"):
                            cats.pop(k, None)
                if istate in ("busy", "away", "passing") and not (cats.keys() & {"strike", "contact"}):
                    cats.pop("pose", None)
                    cats.pop("object_held", None)
                if not directed:
                    cats.pop("audio", None)
                    for k in ("pose", "contact"):
                        cats.pop(k, None)
                hard_act = (directed and bool(cats.keys() & PHYSICAL_ACT)) or (
                    "weapon_held" in cats and (istate == "targeted" or it["aimed"] or dist <= WEAPON_DIRECT_DIST_M))

                fctx = {**actx, "dark": scene_ctx["dark"], "safe": scene_ctx["safe"], "weapon_label": obj_label}
                rank, score, reason = fuse(cats, fctx)
                if scene_ctx["safe"]:
                    hard = {k: v for k, v in cats.items() if k in HARD_AT_HOME}
                    if hard:
                        r2, s2, why2 = fuse(hard, {**fctx, "safe": False})
                        if min(r2, 1) > rank:
                            rank, score, reason = min(r2, 1), max(score, min(s2, 0.6)), why2

                st["near_since"] = (st["near_since"] or ft) if dist <= 3.5 else None
                near_s = ft - st["near_since"] if st["near_since"] else 0.0
                rider, pillion = st["rider"]
                local = pre_attack_cues(dist=dist, approach=approach, isolated=isolated, dark=bool(scene_ctx["dark"]),
                                        glances=glances, scans=scans, reapproach=episodes,
                                        two_wheeler=rider == "two_wheeler", pillion=pillion, near_s=near_s,
                                        company=company)
                if istate not in LOCAL_CUE_STATES and follow_lvl == 0:
                    local = []
                local_hot = [c for c in local if c[0] >= 1]
                for r_, _tag, why_ in local_hot:
                    if r_ > rank:
                        rank, score, reason = r_, max(score, 0.45), why_
                if rank >= 2 and not hard_act:
                    rank, score = 1, min(score, 0.69)
                raw_rank = rank
                if rank >= 1 and wall - st["last_dbg"] > 1.0:
                    st["last_dbg"] = wall
                    log.info("[intent] %s %s attn=%s busy=%.2f aimed=%s miss=%s %s", key, istate, it["attn"],
                             it["busy"], it["aimed"], it["miss_m"], it["why"])
                    log.info("[threat] %s rank=%d d=%.1fm appr=%.2f cats=%s local=%s iso=%s crowd=%s dark=%s(%.0f) "
                             "safe=%s crowdact=%s | %s", key, rank, dist, approach, sorted(cats),
                             [c[1] for c in local], isolated, actx.get("crowd"), scene_ctx["dark"],
                             scene_ctx.get("brightness", 0.0), scene_ctx["safe"], crowd_active, reason)
                if "weapon_held" in cats:
                    reason = f"Possible {obj_label} in hand" + ("" if rank < 2 else " + " + reason.split(", ", 1)[-1])

                cue_sig = frozenset(cats) | frozenset(c[1] for c in local_hot)

                # 2. collect finished gemini check
                fresh = None
                pt = pending.get(key)
                if pt is not None and pt[0].done():
                    pending.pop(key, None)
                    try:
                        res_v = pt[0].result()
                    except Exception:
                        res_v = None
                    if res_v is not None:
                        verdicts[key] = (ft, *res_v, pt[1])
                        fresh = res_v[0] == "threat" and res_v[1] >= _min_conf(pt[1])
                        if res_v[0] == "none" and res_v[1] >= GEMINI_CLEAR_CONF:
                            cleared[key] = (ft + CLEARED_HOLD_S, cue_sig)
                        else:
                            cleared.pop(key, None)

                # 3. rolling crop buffer for nearby tracks
                if dist <= CROP_MAX_DIST_M and ft - st["last_crop"] >= CROP_INTERVAL_S:
                    crop = _person_crop(frame, d["box"], wide=held)
                    if crop is not None:
                        crop_history.setdefault(key, deque(maxlen=TEMPORAL_FRAME_COUNT)).append((ft, crop))
                        st["last_crop"] = ft
                history = crop_history.get(key)
                ready = [c for ts, c in history if ft - ts <= TEMPORAL_WINDOW_S] if history else []

                # 4. want a gemini check?
                want = rank >= 2 or (rank == 1 and (bool(GEMINI_RANK1_KEYS & set(cats)) or bool(local_hot)))
                scan = (not want and dist <= GEMINI_SCAN_DIST_M
                        and (obj_alert or istate in ("targeted", "watching", "unknown"))
                        and (obj_alert or p_score >= 0.3 or any(c[0] == 0 for c in local)
                             or (isolated and approach >= 0.5)))
                kind = "critical" if rank >= 2 else ("rank1" if want else "scan")
                cd = COOLDOWN_CRITICAL_S if rank >= 2 else (COOLDOWN_MODERATE_S if want else COOLDOWN_SCAN_S)
                clr = cleared.get(key)
                if clr is not None and kind != "critical":
                    if ft >= clr[0]:
                        cleared.pop(key, None)
                    else:
                        new_cues = cue_sig - clr[1]
                        escalated = bool(new_cues & (GEMINI_RANK1_KEYS | {c[1] for c in local_hot}))
                        if kind == "scan" or not escalated:
                            want = scan = False
                if (want or scan) and key not in pending and ft - cooldown.get(key, 0.0) > cd and ready:
                    sensor = "; ".join(s for s in (
                        "nobody else seen for 20 s" if isolated else "",
                        "dark" if scene_ctx["dark"] else "",
                        "at home or a safe zone" if scene_ctx["safe"] else "",
                        ("rider on a two-wheeler" + (" with a pillion" if pillion else "")) if rider == "two_wheeler"
                        else (f"standing at a {rider}" if rider else ""),
                        (f"words heard nearby (speaker not localised): {', '.join(assess['heard'])}"
                         if assess.get("heard") else ""),
                        ) if s)
                    prio = ({"critical": 2, "rank1": 1, "scan": 0}[kind], int(obj_alert), -dist)
                    hint = obj_label if (held or obj_label in SHARP) else ""
                    candidates.append((prio, key, kind, ready, st["last_crop"], d["box"], held, dist, hint, sensor))

                # 5. apply verdict
                v = verdicts.get(key)
                if v and ft - v[0] > VERDICT_TTL:
                    verdicts.pop(key, None)
                    v = None
                verified = bool(v and v[1] == "threat" and v[2] >= _min_conf(v[6]))
                pattern, cleared_now = "", False
                if v and v[1] in ("suspicious", "threat") and not verified and v[2] >= GEMINI_SUSPICIOUS_CONF:
                    if rank < 1:
                        reason = v[3] or reason
                    rank, score, pattern = max(rank, 1), max(score, 0.5), v[5]
                if verified:
                    rank, score, reason, pattern = 2, max(score, 0.9), v[3] or reason, v[5]
                if (GEMINI_CAN_CLEAR and v and v[1] == "none" and v[2] >= GEMINI_CLEAR_CONF and rank == 1
                        and not hard_act):
                    rank, score, reason, cleared_now = 0, min(score, 0.2), "", True
                rank, score, reason = _stabilise(st, rank, score, reason, ft, clear=cleared_now)
                level = RANK_LEVEL[rank]
                cmd = v[4] if verified else ""
                if level == "CRITICAL" and not cmd:
                    cmd = "Threat close. Move away. Shout for help."

                # 6. SOS popup arming
                crit_in_range = level == "CRITICAL" and dist <= AUTO_SOS_MAX_DIST_M
                sos_grade = bool(verified and crit_in_range and v[2] >= AUTO_SOS_MIN_CONF
                                 and (hard_act or v[5] in SOS_PATTERNS))
                local_grade = bool(AUTO_SOS_ON_CRITICAL and crit_in_range)
                if crit_in_range:
                    last_critical_ts, last_critical_reason = wall, reason
                    if countdown_task is None:
                        armed_frame = frame          # reference, not a 1-6 MB copy per frame
                if (sos_grade and fresh) or (local_grade and wall - st["crit_hit"] >= AUTO_SOS_CRIT_SPACING_S):
                    st["crit_hit"] = wall
                    maybe_arm(key, wall, frame, d["box"], reason, {
                        "track_id": key, "pattern": v[5] if v else "local",
                        "confidence": round(v[2], 2) if v else round(score, 2),
                        "distance_m": dist, "approach_mps": round(approach, 2), "intent": istate,
                        "object": obj_label, "hard_act": hard_act, "instruction": cmd, "reason": reason,
                    })
                elif not crit_in_range and fresh is False:
                    confirmations.pop(key, None)
                if level == "CRITICAL" and countdown_task is None and wall - st["sos_dbg"] > 2.0:
                    st["sos_dbg"] = wall
                    log.info("[sos] %s CRITICAL: in_range=%s sos_grade=%s local_grade=%s verified=%s conf=%s "
                             "hard_act=%s dist=%.1f hits=%d busy=%s recent_fire=%s snoozed=%s", key, crit_in_range,
                             sos_grade, local_grade, verified, v[2] if v else None, hard_act, dist,
                             len(confirmations.get(key, [])), fire_busy(), recently_fired(wall),
                             wall < snoozed.get(key, 0.0))

                # 6b. persist events with their cue vector
                if rank >= 1 and (rank >= 2 or LOG_MODERATE):
                    rec = logged.get(key)
                    cool = THREAT_LOG_COOLDOWN_S if rank >= 2 else MODERATE_LOG_COOLDOWN_S
                    if rec is None or wall - rec[0] > cool or rank > rec[1] or (verified and not rec[2]):
                        logged[key] = (wall, rank, verified)
                        feats = {
                            "cats": sorted(cats), "local": sorted(c[1] for c in local), "raw_rank": raw_rank,
                            "pose": p_score, "pose_why": p_why, "dist": dist, "approach": round(approach, 2),
                            "dwell": assess["dwell_s"], "isolated": isolated, "crowd": actx.get("crowd"),
                            "dark": bool(scene_ctx["dark"]), "safe": bool(scene_ctx["safe"]), "company": company,
                            "crowd_activity": crowd_active, "object": obj_label, "held": held, "raised": raised,
                            "rider": rider, "pillion": pillion, "glances": glances, "scans": scans,
                            "reapproach": episodes, "follow": follow_lvl, "user_speed": round(user_speed, 2),
                            "intent": istate, "attn": it["attn"], "busy": it["busy"], "aimed": it["aimed"],
                            "miss_m": it["miss_m"], "hard_act": hard_act, "sos_grade": sos_grade,
                            "local_grade": local_grade,
                            "gemini": ({"sev": v[1], "conf": round(v[2], 2), "pattern": v[5], "kind": v[6]}
                                       if v else None),
                        }
                        spawn(log_threat_event(user.id, key, level, "VIDEO", score, reason, dist, approach,
                                               verified, gps, recent_events, feats))

                # 7. HUD relevance
                if (rank >= 1 or verified or key in pending
                        or (SEND_NORMAL_BOXES and level == "NORMAL" and dist <= HUD_NORMAL_MAX_DIST_M)):
                    targets.append(ThreatBox(
                        id=key, level=level, threat_score=float(min(1.0, max(0.0, score))),
                        approach_mps=approach, dwell_s=int(assess["dwell_s"]), verified_threat=verified,
                        verification_pending=key in pending,
                        threat_reasoning=reason if level != "NORMAL" else "", unity_instruction=cmd,
                        pattern=pattern, distance_m=dist, intent=istate,
                        attention=-1.0 if it["attn"] is None else float(it["attn"]),
                        vx=d["vx"], vy=d["vy"],
                        bbox=BoundaryBox(x_min=x1 / w_img, y_min=y1 / h_img,
                                         width=min(1.0, bw / w_img), height=min(1.0, bh / h_img)),
                    ))

            # ---- D. dispatch gemini checks, most urgent first, within budget ----
            for _prio, key, kind, ready, last_ts, box, wide, dist, hint, sensor in sorted(
                    candidates, key=lambda c: c[0], reverse=True):
                now_w = time.time()
                if len(pending) >= MAX_GEMINI_INFLIGHT or not _gemini_ok(now_w):
                    break
                if not _budget_ok(user.id, now_w):
                    if now_w - last_budget_log > 300:
                        last_budget_log = now_w
                        log.warning("Gemini budget reached (user=%s): local detection only", user.id)
                    break
                while my_calls and now_w - my_calls[0] > 60.0:
                    my_calls.popleft()
                if len(my_calls) >= (GEMINI_USER_RPM * 2 if kind == "critical" else GEMINI_USER_RPM):
                    continue
                if kind == "scan" and not hint and now_w - last_scan < SCAN_GAP_S:
                    continue
                n_calls = len(_gemini_calls)
                if (kind == "rank1" and n_calls >= GEMINI_RPM * GEMINI_RANK1_BUDGET) or \
                        (kind == "scan" and n_calls >= GEMINI_RPM * GEMINI_SCAN_BUDGET):
                    continue
                crops = list(ready[-TEMPORAL_FRAME_COUNT:])
                if ft - last_ts > TRIGGER_CROP_AGE_S:
                    trig = _person_crop(frame, box, wide)
                    if trig is not None:
                        crops = crops[-max(1, TEMPORAL_FRAME_COUNT - 1):] + [trig]
                if len(crops) < TEMPORAL_MIN_FRAMES:
                    continue
                _gemini_calls.append(now_w)
                my_calls.append(now_w)
                _budget_spend(user.id, now_w)
                if kind == "scan":
                    last_scan = now_w
                pending[key] = (asyncio.create_task(gemini_check(crops, dist, hint, sensor, frame, box)), kind)
                cooldown[key] = ft

            # ---- E. forget tracks that disappeared ----
            engine.drop_missing(seen, ft)
            gallery.prune(ft)
            follow_test.prune(ft)
            for k in seen:
                last_seen[k] = ft
            for k in [k for k, ts in last_seen.items() if ft - ts > TRACK_TTL_S]:
                last_seen.pop(k, None)
                pt = pending.pop(k, None)
                if pt is not None:
                    pt[0].cancel()
                for dct in (verdicts, crop_history, tstate, confirmations, cooldown, logged, snoozed, cleared):
                    dct.pop(k, None)

            targets.sort(key=lambda tg: (tg.verified_threat, tg.level == "CRITICAL",
                                         tg.verification_pending, tg.threat_score), reverse=True)
            if MAX_UI_TARGETS > 0:
                targets = targets[:MAX_UI_TARGETS]
            await send_json(DataPayload(timestamp_ms=int(wall * 1000),
                                        frame_latency_ms=int((time.time() - recv_ts) * 1000), targets=targets))

            # ---- audio-only alert ----
            now_w = time.time()
            audio_hits[:] = [s for s in audio_hits if now_w - s <= AUDIO_ALERT_WINDOW_S]
            if len(audio_hits) >= AUDIO_ALERT_HITS and now_w - audio_alert_sent >= AUDIO_ALERT_REPEAT_S:
                audio_alert_sent = now_w
                await send_json({"type": "audio_alert", "scream": round(scream_val, 2),
                                 "unity_instruction": "Scream detected. Stay alert."})
                spawn(log_threat_event(user.id, "audio", "CRITICAL", "AUDIO", scream_val,
                                       "Sustained scream detected", None, None, False, gps, recent_events))
                if can_prompt(now_w, "audio"):
                    armed_track = "audio"
                    armed_frame, armed_reason = frame, "Sustained scream detected"
                    armed_details = {"source": "audio", "pattern": "scream", "confidence": round(scream_val, 2),
                                     "reason": "Sustained scream detected", "thumb_b64": None, **gps_details()}
                    countdown_task = asyncio.create_task(countdown())

    except WebSocketDisconnect:
        pass
    except Exception as e:
        if not closed:
            log.exception("WebSocket error (%s): %s", user.id, e)
    finally:
        _active_connections -= 1
        closed = True
        for t in (recv_task, infer_task, obj_task):
            if t is not None:
                t.cancel()
        for pt in pending.values():
            pt[0].cancel()

        now_f = time.time()
        if countdown_task and not countdown_task.done():
            countdown_task.cancel()
            if not fire_busy():
                log.warning("dead-man switch (popup open): firing SOS for %s", user.id)
                start_fire(f"Connection lost during SOS popup: {armed_reason}")
        elif (AUTO_SOS_ENABLED and not fire_busy() and not recently_fired(now_f)
              and now_f - last_critical_ts <= DEADMAN_RECENT_CRITICAL_S):
            log.warning("dead-man switch (recent critical): firing SOS for %s", user.id)
            start_fire(f"Connection lost right after critical threat: {last_critical_reason}")

        try:
            await websocket.close()
        except Exception:
            pass