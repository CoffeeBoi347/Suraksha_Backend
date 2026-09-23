import asyncio
import logging
import os
import secrets
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone

import httpx
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field, field_validator

from app.auth import get_current_user
from app.db import User, db, normalize_in_phone, supabase_admin

log = logging.getLogger("suraksha.sos")
router = APIRouter(prefix="/sos", tags=["SOS Emergency Engine"])

# ---------- config ----------
FAST2SMS_API_KEY = os.getenv("FAST2SMS_API_KEY")
FAST2SMS_URL = os.getenv("FAST2SMS_URL", "https://www.fast2sms.com/dev/bulkV2")
# "dlt" = DLT-registered template (required for reliable delivery with links in India)
# "q"   = quick route (fine for testing; links may be blocked by TRAI URL whitelisting)
FAST2SMS_ROUTE = os.getenv("FAST2SMS_ROUTE", "q")
FAST2SMS_SENDER_ID = os.getenv("FAST2SMS_SENDER_ID", "")
FAST2SMS_DLT_TEMPLATE_ID = os.getenv("FAST2SMS_DLT_TEMPLATE_ID", "")  # vars: name|lat,lng|track_url

EXOTEL_SID = os.getenv("EXOTEL_ACCOUNT_SID")
EXOTEL_CALLER_ID = os.getenv("EXOTEL_CALLER_ID")
EXOTEL_APP_ID = os.getenv("EXOTEL_APP_ID")
EXOTEL_API_KEY = os.getenv("EXOTEL_API_KEY")
EXOTEL_API_TOKEN = os.getenv("EXOTEL_API_TOKEN")
EXOTEL_SUBDOMAIN = os.getenv("EXOTEL_SUBDOMAIN", "api.exotel.com")  # api.in.exotel.com for Mumbai cluster
EXOTEL_WEBHOOK_SECRET = os.getenv("EXOTEL_WEBHOOK_SECRET", "")      # append ?key=... to the Passthru URL

# Tracking page on YOUR domain — whitelist this domain with your DLT provider.
TRACK_BASE_URL = os.getenv("TRACK_BASE_URL", "https://suraksha-app.com/t")

NATIONAL_EMERGENCY = "112"
MAX_CONTACTS = 5
DEDUPE_WINDOW = timedelta(minutes=30)
RATE_LIMIT_N, RATE_LIMIT_WINDOW = 5, 600  # 5 triggers / 10 min / user
RETRY_DELAYS = (0, 2, 5, 12)              # 4 attempts

_trigger_log: dict[str, deque] = defaultdict(deque)


# ---------- models ----------
class SOSEmergencyRequest(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    state_code: str = Field("NATIONAL", max_length=16)
    district: str = Field("All", max_length=64)
    accuracy_m: float = Field(5.0, ge=0)
    speed_mps: float = Field(0.0, ge=0)
    battery_percentage: int = Field(..., ge=0, le=100)
    snapshot_url: str | None = Field(None, max_length=500)
    threat_level: str = Field("CRITICAL", pattern="^(LOW|MEDIUM|HIGH|CRITICAL)$")


class LocationUpdate(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    accuracy_m: float = Field(5.0, ge=0)
    speed_mps: float = Field(0.0, ge=0)
    battery_percentage: int | None = Field(None, ge=0, le=100)


class CancelRequest(BaseModel):
    reason: str = Field("USER_SAFE", pattern="^(USER_SAFE|FALSE_ALARM|RESOLVED)$")
    notify_contacts: bool = True


class ContactRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=60)
    phone_number: str

    @field_validator("phone_number")
    @classmethod
    def _phone(cls, v: str) -> str:
        return normalize_in_phone(v)


# ---------- helpers ----------
def _now() -> datetime:
    return datetime.now(timezone.utc)


def _rate_limited(user_id: str) -> bool:
    q, t = _trigger_log[user_id], time.time()
    while q and t - q[0] > RATE_LIMIT_WINDOW:
        q.popleft()
    if len(q) >= RATE_LIMIT_N:
        return True
    q.append(t)
    return False


def _is_short_code(num: str) -> bool:
    return len("".join(c for c in num if c.isdigit())) < 8


def _ten_digit(num: str) -> str:
    d = "".join(c for c in num if c.isdigit())
    return d[-10:]


def _track_url(share_token: str) -> str:
    return f"{TRACK_BASE_URL}/{share_token}"


async def _update_incident(incident_id: str, fields: dict):
    try:
        await db(lambda: supabase_admin.table("sos_incidents").update(fields).eq("id", incident_id).execute())
    except Exception as e:
        log.error("incident %s update failed: %s", incident_id, e)


async def _log_dispatch(incident_id: str, channel: str, target: str, ok: bool, attempt: int, detail: str):
    try:
        await db(lambda: supabase_admin.table("sos_dispatch_log").insert({
            "incident_id": incident_id, "channel": channel, "target": target,
            "success": ok, "attempt": attempt, "detail": detail[:500],
        }).execute())
    except Exception as e:
        log.error("dispatch log failed: %s", e)


# ---------- providers ----------
async def _fast2sms_send(numbers: list[str], text: str, dlt_vars: str) -> tuple[bool, str]:
    if not FAST2SMS_API_KEY:
        return False, "FAST2SMS_API_KEY missing"
    nums = ",".join(_ten_digit(n) for n in numbers)
    if FAST2SMS_ROUTE == "dlt" and FAST2SMS_DLT_TEMPLATE_ID:
        data = {"route": "dlt", "sender_id": FAST2SMS_SENDER_ID, "message": FAST2SMS_DLT_TEMPLATE_ID,
                "variables_values": dlt_vars, "numbers": nums, "flash": "0"}
    else:
        data = {"route": "q", "message": text, "numbers": nums, "flash": "0"}

    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.post(FAST2SMS_URL, headers={"authorization": FAST2SMS_API_KEY}, data=data)
    try:
        body = r.json()
    except ValueError:
        return False, f"HTTP {r.status_code}: {r.text[:200]}"
    if body.get("return"):
        return True, f"request_id={body.get('request_id')}"
    return False, str(body.get("message"))


async def _exotel_call(target_phone: str, incident_id: str) -> tuple[bool, str]:
    if not all([EXOTEL_SID, EXOTEL_API_KEY, EXOTEL_API_TOKEN, EXOTEL_CALLER_ID, EXOTEL_APP_ID]):
        return False, "Exotel credentials missing"
    url = f"https://{EXOTEL_SUBDOMAIN}/v1/Accounts/{EXOTEL_SID}/Calls/connect.json"
    data = {
        "From": "0" + _ten_digit(target_phone),
        "CallerId": EXOTEL_CALLER_ID,
        "Url": f"http://my.exotel.com/{EXOTEL_SID}/exoml/start_voice/{EXOTEL_APP_ID}",
        "CallType": "trans",
        "TimeLimit": "300",
        "CustomField": incident_id,  # TTS endpoint looks the incident up by id — no PII in transit
    }
    async with httpx.AsyncClient(timeout=10.0, auth=(EXOTEL_API_KEY, EXOTEL_API_TOKEN)) as client:
        r = await client.post(url, data=data)
    if r.status_code == 200:
        try:
            sid = r.json().get("Call", {}).get("Sid")
        except ValueError:
            sid = None
        return True, f"call_sid={sid}"
    return False, f"HTTP {r.status_code}: {r.text[:200]}"


async def _with_retry(incident_id: str, channel: str, target: str, fn) -> bool:
    for attempt, delay in enumerate(RETRY_DELAYS, start=1):
        if delay:
            await asyncio.sleep(delay)
        try:
            ok, detail = await fn()
        except Exception as e:
            ok, detail = False, f"exception: {e}"
        await _log_dispatch(incident_id, channel, target, ok, attempt, detail)
        if ok:
            return True
        log.warning("[%s] %s attempt %d failed: %s", incident_id, channel, attempt, detail)
    log.error("[%s] %s FAILED after %d attempts", incident_id, channel, len(RETRY_DELAYS))
    return False


# ---------- dispatch worker ----------
async def dispatch_incident(incident_id: str):
    """Idempotent: only runs channels still PENDING/FAILED. Safe to re-run after a crash."""
    try:
        res = await db(lambda: supabase_admin.table("sos_incidents").select("*")
                       .eq("id", incident_id).single().execute())
        inc = res.data
    except Exception as e:
        log.error("dispatch: cannot load incident %s: %s", incident_id, e)
        return
    if not inc or inc["status"] != "ACTIVE":
        return

    name = inc.get("user_name") or "A Suraksha user"
    lat, lng = inc["latitude"], inc["longitude"]
    link = _track_url(inc["share_token"])
    loc = f"{lat:.5f},{lng:.5f}" if lat is not None and lng is not None else "unknown (updating)"
    auto = " (auto-detected threat)" if inc.get("trigger_source") == "AUTO_HUD" else ""
    text = (f"SURAKSHA SOS{auto}: {name} needs help NOW. "
            f"Location {loc}. Live: {link} "
            f"Call them or dial 112.")
    dlt_vars = f"{name}|{loc}|{link}"

    jobs = []

    # contacts
    if inc["contacts_status"] in ("PENDING", "FAILED"):
        c = await db(lambda: supabase_admin.table("emergency_contacts").select("phone_number")
                     .eq("user_id", inc["user_id"]).execute())
        numbers = [x["phone_number"] for x in (c.data or [])]
        if numbers:
            await _update_incident(incident_id, {"contacts_status": "SENDING"})

            async def contacts_job():
                ok = await _with_retry(incident_id, "SMS_CONTACTS", ",".join(numbers),
                                       lambda: _fast2sms_send(numbers, text, dlt_vars))
                await _update_incident(incident_id, {"contacts_status": "SENT" if ok else "FAILED"})
            jobs.append(contacts_job())
        else:
            await _update_incident(incident_id, {"contacts_status": "NO_CONTACTS"})

    # official desk
    desk_phone = inc.get("dispatched_phone") or NATIONAL_EMERGENCY
    if inc["desk_status"] in ("PENDING", "FAILED"):
        if _is_short_code(desk_phone):
            await _update_incident(incident_id, {"desk_status": "CLIENT_DIALS"})
        elif inc.get("desk_channel") == "SMS":
            await _update_incident(incident_id, {"desk_status": "SENDING"})

            async def desk_sms_job():
                ok = await _with_retry(incident_id, "SMS_DESK", desk_phone,
                                       lambda: _fast2sms_send([desk_phone], "[DISPATCH] " + text, dlt_vars))
                await _update_incident(incident_id, {"desk_status": "SENT" if ok else "FAILED"})
            jobs.append(desk_sms_job())
        else:
            await _update_incident(incident_id, {"desk_status": "SENDING"})

            async def desk_call_job():
                ok = await _with_retry(incident_id, "IVR_DESK", desk_phone,
                                       lambda: _exotel_call(desk_phone, incident_id))
                await _update_incident(incident_id, {"desk_status": "SENT" if ok else "FAILED"})
            jobs.append(desk_call_job())

    if jobs:
        await asyncio.gather(*jobs, return_exceptions=True)


async def resume_pending_dispatches():
    """Call on startup: re-dispatch anything a crash/redeploy left half-done."""
    cutoff = (_now() - DEDUPE_WINDOW).isoformat()
    try:
        res = await db(lambda: supabase_admin.table("sos_incidents").select("id")
                       .eq("status", "ACTIVE").gte("created_at", cutoff)
                       .or_("contacts_status.in.(PENDING,SENDING,FAILED),desk_status.in.(PENDING,SENDING,FAILED)")
                       .execute())
    except Exception as e:
        log.error("resume scan failed: %s", e)
        return
    for row in res.data or []:
        # SENDING at boot means the worker died mid-flight → retry
        await db(lambda rid=row["id"]: supabase_admin.table("sos_incidents")
                 .update({"contacts_status": "FAILED"}).eq("id", rid).eq("contacts_status", "SENDING").execute())
        await db(lambda rid=row["id"]: supabase_admin.table("sos_incidents")
                 .update({"desk_status": "FAILED"}).eq("id", rid).eq("desk_status", "SENDING").execute())
        asyncio.create_task(dispatch_incident(row["id"]))
    if res.data:
        log.warning("resumed %d pending SOS dispatches", len(res.data))


async def _find_desk(state_code: str, district: str) -> dict:
    fallback = {"organization_name": "Pan-India Emergency Response (112)",
                "phone_number": NATIONAL_EMERGENCY, "can_receive_sms": False}
    t = supabase_admin.table("safety_desks")
    try:
        for q in (
            lambda: t.select("*").eq("state_code", state_code).eq("district", district).eq("is_active", True).limit(1).execute(),
            lambda: t.select("*").eq("state_code", state_code).eq("is_active", True).limit(1).execute(),
            lambda: t.select("*").eq("state_code", "NATIONAL").eq("is_active", True).limit(1).execute(),
        ):
            r = await db(q)
            if r.data:
                return r.data[0]
    except Exception as e:
        log.warning("desk lookup failed: %s", e)
    return fallback


def _public_incident(inc: dict) -> dict:
    return {
        "incident_id": inc["id"],
        "status": inc["status"],
        "track_url": _track_url(inc["share_token"]),
        "maps_link": (f"https://maps.google.com/?q={inc['latitude']},{inc['longitude']}"
                      if inc.get("latitude") is not None else None),
        "contacts_status": inc.get("contacts_status"),
        "desk_status": inc.get("desk_status"),
        "official_desk": inc.get("dispatched_to_desk"),
        # Client MUST open the dialer with this (tel:112). Backend can't reach emergency short codes.
        "client_should_dial": NATIONAL_EMERGENCY,
    }


async def _get_owned_incident(incident_id: str, user_id: str) -> dict:
    r = await db(lambda: supabase_admin.table("sos_incidents").select("*")
                 .eq("id", incident_id).eq("user_id", user_id).maybe_single().execute())
    if not r or not r.data:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Incident not found")
    return r.data


# ---------- routes ----------
_bg_tasks: set[asyncio.Task] = set()


def spawn(coro) -> asyncio.Task:
    """Fire-and-forget that survives the caller (request/WebSocket) ending."""
    t = asyncio.create_task(coro)
    _bg_tasks.add(t)
    t.add_done_callback(_bg_tasks.discard)
    return t


class SOSFireError(Exception):
    def __init__(self, code: int, detail):
        self.code, self.detail = code, detail


async def fire_sos(user: User, *, latitude: float | None, longitude: float | None,
                   accuracy_m: float = 0.0, speed_mps: float = 0.0, battery_percentage: int | None = None,
                   state_code: str = "NATIONAL", district: str = "All", snapshot_url: str | None = None,
                   threat_level: str = "CRITICAL", source: str = "MANUAL") -> dict:
    """Single entry point for every SOS (button, voice, auto-trigger from the HUD)."""
    uid = user.id

    # 1. dedupe: repeated taps / auto + manual overlap → same incident
    since = (_now() - DEDUPE_WINDOW).isoformat()
    try:
        existing = await db(lambda: supabase_admin.table("sos_incidents").select("*")
                            .eq("user_id", uid).eq("status", "ACTIVE").gte("created_at", since)
                            .order("created_at", desc=True).limit(1).execute())
        existing_rows = existing.data or []
    except Exception as e:
        log.error("dedupe lookup failed: %s", e)
        existing_rows = []
    if existing_rows:
        inc = existing_rows[0]
        if latitude is not None and longitude is not None:
            await _record_location(inc["id"], latitude, longitude, accuracy_m, speed_mps, battery_percentage)
        if snapshot_url and not inc.get("snapshot_url"):
            await _update_incident(inc["id"], {"snapshot_url": snapshot_url})
        if inc["contacts_status"] == "FAILED" or inc["desk_status"] == "FAILED":
            spawn(dispatch_incident(inc["id"]))
        return {**_public_incident(inc), "deduplicated": True}

    if _rate_limited(uid):
        raise SOSFireError(status.HTTP_429_TOO_MANY_REQUESTS, "Too many SOS triggers. Dial 112 directly.")

    # 2. persist first
    meta = user.user_metadata or {}
    desk = await _find_desk(state_code.upper(), district)
    incident = {
        "user_id": uid,
        "user_name": (meta.get("full_name") or "A Suraksha user")[:60],
        "share_token": secrets.token_urlsafe(12),
        "latitude": latitude,
        "longitude": longitude,
        "accuracy_m": accuracy_m,
        "speed_mps": speed_mps,
        "battery_percentage": battery_percentage,
        "snapshot_url": snapshot_url,
        "threat_level": threat_level,
        "trigger_source": source,
        "dispatched_to_desk": desk.get("organization_name"),
        "dispatched_phone": desk.get("phone_number"),
        "desk_channel": "SMS" if desk.get("can_receive_sms") else "IVR",
        "contacts_status": "PENDING",
        "desk_status": "PENDING",
        "status": "ACTIVE",
    }
    try:
        r = await db(lambda: supabase_admin.table("sos_incidents").insert(incident).execute())
        inc = r.data[0]
    except Exception as e:
        log.critical("SOS INSERT FAILED for %s: %s", uid, e)
        raise SOSFireError(status.HTTP_503_SERVICE_UNAVAILABLE,
                           {"message": "SOS server error. Dial 112 now.", "client_should_dial": "112"})

    if latitude is not None and longitude is not None:
        await _record_location(inc["id"], latitude, longitude, accuracy_m, speed_mps, battery_percentage)

    # 3. dispatch
    spawn(dispatch_incident(inc["id"]))
    log.warning("SOS FIRED user=%s incident=%s source=%s", uid, inc["id"], source)
    return {**_public_incident(inc), "deduplicated": False}


@router.post("/trigger", status_code=status.HTTP_200_OK)
async def trigger_sos(payload: SOSEmergencyRequest, current_user: User = Depends(get_current_user)):
    try:
        return await fire_sos(current_user, **payload.model_dump(), source="MANUAL")
    except SOSFireError as e:
        raise HTTPException(e.code, e.detail)


async def _record_location(incident_id: str, lat: float, lng: float, acc: float, spd: float, batt: int | None):
    row = {"incident_id": incident_id, "latitude": lat, "longitude": lng, "accuracy_m": acc, "speed_mps": spd}
    if batt is not None:
        row["battery_percentage"] = batt
    upd = {"latitude": lat, "longitude": lng, "accuracy_m": acc, "speed_mps": spd, "last_seen_at": _now().isoformat()}
    if batt is not None:
        upd["battery_percentage"] = batt
    try:
        await db(lambda: supabase_admin.table("sos_location_updates").insert(row).execute())
        await db(lambda: supabase_admin.table("sos_incidents").update(upd).eq("id", incident_id).execute())
    except Exception as e:
        log.error("location record failed for %s: %s", incident_id, e)


@router.post("/{incident_id}/location")
async def update_location(incident_id: str, payload: LocationUpdate,
                          current_user: User = Depends(get_current_user)):
    """Phone calls this every 10–15 s while an SOS is active."""
    inc = await _get_owned_incident(incident_id, current_user.id)
    if inc["status"] != "ACTIVE":
        raise HTTPException(status.HTTP_409_CONFLICT, "Incident is not active")
    await _record_location(incident_id, payload.latitude, payload.longitude,
                           payload.accuracy_m, payload.speed_mps, payload.battery_percentage)
    return {"ok": True}


@router.post("/{incident_id}/cancel")
async def cancel_sos(incident_id: str, payload: CancelRequest, background_tasks: BackgroundTasks,
                     current_user: User = Depends(get_current_user)):
    inc = await _get_owned_incident(incident_id, current_user.id)
    if inc["status"] != "ACTIVE":
        return {"ok": True, "status": inc["status"]}
    await _update_incident(incident_id, {"status": payload.reason, "closed_at": _now().isoformat()})

    if payload.notify_contacts and inc.get("contacts_status") in ("SENT", "SENDING"):
        async def notify_safe():
            c = await db(lambda: supabase_admin.table("emergency_contacts").select("phone_number")
                         .eq("user_id", current_user.id).execute())
            nums = [x["phone_number"] for x in (c.data or [])]
            if nums:
                name = inc.get("user_name") or "Your contact"
                msg = f"SURAKSHA UPDATE: {name} has marked themselves SAFE. SOS cancelled."
                await _with_retry(incident_id, "SMS_SAFE", ",".join(nums),
                                  lambda: _fast2sms_send(nums, msg, f"{name}|SAFE|-"))
        background_tasks.add_task(notify_safe)
    return {"ok": True, "status": payload.reason}


@router.get("/track/{share_token}")
async def public_track(share_token: str):
    """Backs the tracking page at TRACK_BASE_URL/<token>. Token is unguessable; no auth."""
    r = await db(lambda: supabase_admin.table("sos_incidents")
                 .select("id, user_name, status, latitude, longitude, accuracy_m, battery_percentage, last_seen_at, created_at")
                 .eq("share_token", share_token).maybe_single().execute())
    if not r or not r.data:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Not found")
    inc = r.data
    trail = []
    if inc["status"] == "ACTIVE":
        t = await db(lambda: supabase_admin.table("sos_location_updates")
                     .select("latitude, longitude, created_at").eq("incident_id", inc["id"])
                     .order("created_at", desc=True).limit(50).execute())
        trail = t.data or []
    inc.pop("id")
    return {**inc, "trail": trail if inc["status"] == "ACTIVE" else []}


@router.get("/tts-message", response_class=PlainTextResponse, include_in_schema=False)
async def exotel_tts(CustomField: str = "", key: str = Query("")):
    """
    Exotel Greeting applet → Dynamic URL. Set it to:
      https://<your-api>/sos/tts-message?key=<EXOTEL_WEBHOOK_SECRET>
    Exotel appends CallSid, CustomField, etc. Must return PLAIN TEXT.
    """
    if not EXOTEL_WEBHOOK_SECRET or not secrets.compare_digest(key, EXOTEL_WEBHOOK_SECRET):
        return PlainTextResponse("Unauthorized", status_code=401)

    inc = None
    if CustomField:
        try:
            r = await db(lambda: supabase_admin.table("sos_incidents")
                         .select("user_name, latitude, longitude, battery_percentage, threat_level")
                         .eq("id", CustomField.strip()).maybe_single().execute())
            inc = r.data if r else None
        except Exception as e:
            log.error("tts lookup failed: %s", e)

    if not inc:
        return ("This is an automated emergency alert from the Suraksha safety app. "
                "A user has triggered an SOS. Location details have been sent by SMS.")

    def spoken(x: float) -> str:  # "22.57264" → "22 point 5 7 2 6 4"
        whole, frac = f"{x:.5f}".split(".")
        return f"{whole} point {' '.join(frac)}"

    msg = (f"This is an automated emergency alert from the Suraksha safety app. "
           f"{inc['user_name']} has triggered an SOS. Threat level {inc['threat_level']}. "
           + (f"Latitude {spoken(inc['latitude'])}. Longitude {spoken(inc['longitude'])}. "
              if inc.get("latitude") is not None else "Location not yet available. ")
           + f"Phone battery {inc['battery_percentage']} percent. ")
    return msg + "Repeating. " + msg


# ---------- contacts ----------
@router.post("/contacts", status_code=status.HTTP_201_CREATED)
async def add_contact(payload: ContactRequest, current_user: User = Depends(get_current_user)):
    existing = await db(lambda: supabase_admin.table("emergency_contacts").select("id, phone_number")
                        .eq("user_id", current_user.id).execute())
    rows = existing.data or []
    if any(r["phone_number"] == payload.phone_number for r in rows):
        raise HTTPException(status.HTTP_409_CONFLICT, "Contact already added")
    if len(rows) >= MAX_CONTACTS:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"Maximum {MAX_CONTACTS} emergency contacts")
    res = await db(lambda: supabase_admin.table("emergency_contacts").insert({
        "user_id": current_user.id, "name": payload.name.strip(), "phone_number": payload.phone_number,
    }).execute())
    return {"message": "Contact added successfully", "data": res.data}


@router.get("/contacts")
async def get_contacts(current_user: User = Depends(get_current_user)):
    res = await db(lambda: supabase_admin.table("emergency_contacts").select("id, name, phone_number, created_at")
                   .eq("user_id", current_user.id).order("created_at").execute())
    return {"contacts": res.data or []}


@router.delete("/contacts/{contact_id}")
async def delete_contact(contact_id: str, current_user: User = Depends(get_current_user)):
    res = await db(lambda: supabase_admin.table("emergency_contacts").delete()
                   .eq("id", contact_id).eq("user_id", current_user.id).execute())
    if not res.data:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Contact not found")
    return {"message": "Contact deleted successfully"}


# Keep this LAST — a catch-all path param would otherwise swallow /contacts, /tts-message, /track/...
@router.get("/{incident_id}")
async def get_incident(incident_id: str, current_user: User = Depends(get_current_user)):
    return _public_incident(await _get_owned_incident(incident_id, current_user.id))
