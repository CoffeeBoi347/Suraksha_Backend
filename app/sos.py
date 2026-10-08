import asyncio
import logging
import os
import secrets
import time
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from app.auth import get_current_user
from app.db import User, db, normalize_in_phone, supabase_admin
from app.whatsapp import WA_SAFE_CONTENT_SID, WA_SOS_CONTENT_SID, wa_ready, wa_send

log = logging.getLogger("suraksha.sos")
router = APIRouter(prefix="/sos", tags=["SOS Emergency Engine"])

# Tracking page on YOUR domain.
TRACK_BASE_URL = os.getenv("TRACK_BASE_URL", "https://suraksha-app.com/t")

NATIONAL_EMERGENCY = "112"
MAX_CONTACTS = 3
DEDUPE_WINDOW = timedelta(minutes=30)

# ---- rate limits (counted from sos_incidents: survive restarts and multiple workers) ----
RATE_LIMIT_ENABLED = os.getenv("SOS_RATE_LIMIT", "1") == "1"   # SOS_RATE_LIMIT=0 while testing
RATE_POLICY = {   # source: (max new incidents per 10 min, per 24 h, min seconds between new incidents)
    "MANUAL":   (3, 10, 10),
    "AUTO_HUD": (2, 6, 60),
}
RETRY_DELAYS = (0, 2, 5, 12)

_loc_last: dict[str, float] = {}   # incident_id -> last accepted location update (throttle)

if not wa_ready(WA_SOS_CONTENT_SID):
    log.error("WhatsApp SOS not configured: need TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, WA_SOS_CONTENT_SID")


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


async def _count_incidents(uid: str, source: str, seconds: int) -> int:
    since = (_now() - timedelta(seconds=seconds)).isoformat()
    r = await db(lambda: supabase_admin.table("sos_incidents").select("id", count="exact")
                 .eq("user_id", uid).eq("trigger_source", source).gte("created_at", since).limit(1).execute())
    return r.count or 0


async def _rate_block(uid: str, source: str) -> tuple[str, int] | None:
    """(reason, retry_after_s) if a NEW incident must be refused, else None. Fails open on our own errors."""
    per10, per_day, gap = RATE_POLICY.get(source, RATE_POLICY["AUTO_HUD"])
    try:
        n_gap, n_10, n_day = await asyncio.gather(
            _count_incidents(uid, source, gap), _count_incidents(uid, source, 600),
            _count_incidents(uid, source, 86400))
    except Exception as e:
        log.error("rate check failed, allowing: %s", e)
        return None
    if n_gap:
        return "Too many SOS triggers in a row.", gap
    if n_10 >= per10:
        return "Too many SOS triggers.", 600
    if n_day >= per_day:
        return "Daily SOS limit reached.", 3600
    return None


def _is_short_code(num: str) -> bool:
    return len("".join(c for c in num if c.isdigit())) < 8


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


async def _wa_fanout(incident_id: str, channel: str, numbers: list[str], content_sid: str, wa_vars: dict) -> bool:
    """One retry loop per number, all in parallel. True if at least one was accepted."""
    res = await asyncio.gather(*(
        _with_retry(incident_id, channel, n, lambda n=n: wa_send(n, content_sid, wa_vars))
        for n in numbers), return_exceptions=True)
    return any(r is True for r in res)


async def dispatch_incident(incident_id: str):
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
    log.warning("[%s] dispatch coords lat=%s lng=%s", incident_id, lat, lng)
    if lat is not None and lng is not None:
        loc = f"{lat:.5f}, {lng:.5f}"
        maps = f"https://www.google.com/maps/search/?api=1&query={lat:.6f},{lng:.6f}"
    else:
        loc, maps = "unknown (updating)", _track_url(inc["share_token"])
    wa_vars = {"1": name, "2": loc, "3": maps}
    configured = wa_ready(WA_SOS_CONTENT_SID)

    jobs = []

    # contacts
    if inc["contacts_status"] in ("PENDING", "FAILED"):
        numbers: list[str] = []
        contacts_lookup_ok = True
        try:
            c = await db(lambda: supabase_admin.table("emergency_contacts").select("phone_number")
                         .eq("user_id", inc["user_id"]).execute())
            numbers = [x["phone_number"] for x in (c.data or [])]
        except Exception as e:
            contacts_lookup_ok = False
            log.error("[%s] contacts lookup failed: %s", incident_id, e)

        if not contacts_lookup_ok:
            await _update_incident(incident_id, {"contacts_status": "FAILED"})
        elif not numbers:
            await _update_incident(incident_id, {"contacts_status": "NO_CONTACTS"})
        elif not configured:
            log.error("[%s] WhatsApp SOS not configured, contacts not notified", incident_id)
            await _update_incident(incident_id, {"contacts_status": "FAILED"})
        else:
            await _update_incident(incident_id, {"contacts_status": "SENDING"})

            async def contacts_job():
                ok = await _wa_fanout(incident_id, "WA_CONTACTS", numbers, WA_SOS_CONTENT_SID, wa_vars)
                await _update_incident(incident_id, {"contacts_status": "SENT" if ok else "FAILED"})
            jobs.append(contacts_job())

    # official desk (must be a WhatsApp-enabled mobile; short codes like 112 are dialled by the client)
    desk_phone = inc.get("dispatched_phone") or NATIONAL_EMERGENCY
    if inc["desk_status"] in ("PENDING", "FAILED"):
        if _is_short_code(desk_phone):
            await _update_incident(incident_id, {"desk_status": "CLIENT_DIALS"})
        elif not configured:
            log.error("[%s] WhatsApp SOS not configured, desk not notified", incident_id)
            await _update_incident(incident_id, {"desk_status": "FAILED"})
        else:
            await _update_incident(incident_id, {"desk_status": "SENDING"})

            async def desk_job():
                ok = await _with_retry(incident_id, "WA_DESK", desk_phone,
                                       lambda: wa_send(desk_phone, WA_SOS_CONTENT_SID, wa_vars))
                await _update_incident(incident_id, {"desk_status": "SENT" if ok else "FAILED"})
            jobs.append(desk_job())

    if jobs:
        await asyncio.gather(*jobs, return_exceptions=True)


async def resume_pending_dispatches():
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
        rid = row["id"]
        # SENDING at boot means the worker died mid-flight → retry
        try:
            await db(lambda rid=rid: supabase_admin.table("sos_incidents")
                     .update({"contacts_status": "FAILED"}).eq("id", rid).eq("contacts_status", "SENDING").execute())
            await db(lambda rid=rid: supabase_admin.table("sos_incidents")
                     .update({"desk_status": "FAILED"}).eq("id", rid).eq("desk_status", "SENDING").execute())
        except Exception as e:
            log.error("resume reset failed for %s: %s", rid, e)
        spawn(dispatch_incident(rid))

    if res.data:
        log.warning("resumed %d pending SOS dispatches", len(res.data))


async def _find_desk(state_code: str, district: str) -> dict:
    fallback = {"organization_name": "Pan-India Emergency Response (112)", "phone_number": NATIONAL_EMERGENCY}
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
        "client_should_dial": NATIONAL_EMERGENCY,
    }


async def _get_owned_incident(incident_id: str, user_id: str) -> dict:
    try:
        r = await db(lambda: supabase_admin.table("sos_incidents").select("*")
                     .eq("id", incident_id).eq("user_id", user_id).maybe_single().execute())
    except Exception as e:
        log.error("incident lookup failed (%s): %s", incident_id, e)
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Incident not found")
    if not r or not r.data:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Incident not found")
    return r.data


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

    uid = user.id

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

    blocked = await _rate_block(uid, source) if RATE_LIMIT_ENABLED else None
    if blocked:
        reason, retry = blocked
        log.warning("SOS rate-limited user=%s source=%s: %s", uid, source, reason)
        raise SOSFireError(status.HTTP_429_TOO_MANY_REQUESTS,
                           {"message": f"{reason} Dial 112 directly.", "client_should_dial": NATIONAL_EMERGENCY,
                            "retry_after_s": retry})

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
        "desk_channel": "WHATSAPP",
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


# ---------- contacts (declared before /{incident_id} routes) ----------
@router.post("/contacts", status_code=status.HTTP_201_CREATED)
async def add_contact(payload: ContactRequest, current_user: User = Depends(get_current_user)):
    try:
        existing = await db(lambda: supabase_admin.table("emergency_contacts").select("id, phone_number")
                            .eq("user_id", current_user.id).execute())
        rows = existing.data or []
        if any(r["phone_number"] == payload.phone_number for r in rows):
            raise HTTPException(status.HTTP_409_CONFLICT, "Contact already added")
        if len(rows) >= MAX_CONTACTS:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, f"Maximum {MAX_CONTACTS} emergency contacts")
        res = await db(lambda: supabase_admin.table("emergency_contacts").insert({
            "user_id": current_user.id,
            "contact_name": payload.name.strip(),
            "phone_number": payload.phone_number,
        }).execute())
        return {"message": "Contact added successfully", "data": res.data}
    except HTTPException:
        raise
    except Exception as e:
        log.exception("add_contact failed for %s: %r", current_user.id, e)
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, f"Contact save failed: {e}")


@router.get("/contacts")
async def get_contacts(current_user: User = Depends(get_current_user)):
    try:
        res = await db(lambda: supabase_admin.table("emergency_contacts")
                       .select("id, name:contact_name, phone_number, created_at")
                       .eq("user_id", current_user.id).order("created_at").execute())
        return {"contacts": res.data or []}
    except Exception as e:
        log.exception("get_contacts failed for %s: %r", current_user.id, e)
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, f"Contact fetch failed: {e}")


@router.delete("/contacts/{contact_id}")
async def delete_contact(contact_id: str, current_user: User = Depends(get_current_user)):
    try:
        res = await db(lambda: supabase_admin.table("emergency_contacts").delete()
                       .eq("id", contact_id).eq("user_id", current_user.id).execute())
    except Exception as e:
        # malformed uuid etc. → treat as not found, but log it
        log.exception("delete_contact failed for %s: %r", current_user.id, e)
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Contact not found")
    if not res.data:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Contact not found")
    return {"message": "Contact deleted successfully"}


# ---------- tracking ----------
@router.get("/track/{share_token}")
async def public_track(share_token: str):
    try:
        r = await db(lambda: supabase_admin.table("sos_incidents")
                     .select("id, user_name, status, latitude, longitude, accuracy_m, battery_percentage, last_seen_at, created_at")
                     .eq("share_token", share_token).maybe_single().execute())
    except Exception as e:
        log.exception("track lookup failed: %r", e)
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Tracking unavailable")
    if not r or not r.data:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Not found")
    inc = r.data
    trail = []
    if inc["status"] == "ACTIVE":
        try:
            t = await db(lambda: supabase_admin.table("sos_location_updates")
                         .select("latitude, longitude, created_at").eq("incident_id", inc["id"])
                         .order("created_at", desc=True).limit(50).execute())
            trail = t.data or []
        except Exception as e:
            log.error("trail lookup failed: %s", e)
    inc.pop("id")
    return {**inc, "trail": trail if inc["status"] == "ACTIVE" else []}


# ---------- incident routes ----------
@router.post("/{incident_id}/location")
async def update_location(incident_id: str, payload: LocationUpdate,
                          current_user: User = Depends(get_current_user)):

    inc = await _get_owned_incident(incident_id, current_user.id)
    if inc["status"] != "ACTIVE":
        raise HTTPException(status.HTTP_409_CONFLICT, "Incident is not active")
    now = time.time()
    if now - _loc_last.get(incident_id, 0.0) < 1.0:
        return {"ok": True, "throttled": True}
    if len(_loc_last) > 5000:
        _loc_last.clear()
    _loc_last[incident_id] = now
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
            if not wa_ready(WA_SAFE_CONTENT_SID):
                log.warning("[%s] WA_SAFE_CONTENT_SID not set, contacts not told SAFE", incident_id)
                return
            try:
                c = await db(lambda: supabase_admin.table("emergency_contacts").select("phone_number")
                             .eq("user_id", current_user.id).execute())
                nums = [x["phone_number"] for x in (c.data or [])]
                if nums:
                    name = inc.get("user_name") or "Your contact"
                    await _wa_fanout(incident_id, "WA_SAFE", nums, WA_SAFE_CONTENT_SID, {"1": name})
            except Exception as e:
                log.error("notify_safe failed for %s: %s", incident_id, e)
        background_tasks.add_task(notify_safe)
    return {"ok": True, "status": payload.reason}


# Keep this LAST — a catch-all path param would otherwise swallow /contacts, /track/...
@router.get("/{incident_id}")
async def get_incident(incident_id: str, current_user: User = Depends(get_current_user)):
    return _public_incident(await _get_owned_incident(incident_id, current_user.id))