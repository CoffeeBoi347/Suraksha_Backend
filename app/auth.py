import asyncio
import hashlib
import logging
import os
import secrets
import smtplib
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from pathlib import Path

from dotenv import load_dotenv
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, EmailStr, Field, field_validator

load_dotenv(Path(__file__).resolve().parents[1] / ".env")  # must run before os.getenv below

from app.db import User, db, fresh_client, normalize_in_phone, supabase_admin, user_client
from app.whatsapp import WA_OTP_CONTENT_SID, wa_ready, wa_send

log = logging.getLogger("suraksha.auth")
security = HTTPBearer()
router = APIRouter(prefix="/auth", tags=["Authentication"])

RESET_REDIRECT_URL = os.getenv("RESET_REDIRECT_URL", "https://suraksha-app.com/reset-password")

# OTP delivery: "email" (SMTP, default) or "whatsapp" (Twilio Authentication template, needs Meta business verification)
OTP_CHANNEL = os.getenv("OTP_CHANNEL", "email").strip().lower()
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))          # 587 = STARTTLS, 465 = SSL
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
SMTP_FROM = os.getenv("SMTP_FROM") or SMTP_USER
SMTP_FROM_NAME = os.getenv("SMTP_FROM_NAME", "Suraksha")

OTP_SECRET = os.getenv("OTP_SECRET")
DEV_OTP_CODE = os.getenv("DEV_OTP_CODE")  # DEV ONLY: fixed OTP, nothing sent
OTP_TTL = timedelta(minutes=5)
OTP_RESEND_COOLDOWN = timedelta(seconds=60)
MAX_ATTEMPTS = 5

if DEV_OTP_CODE:
    log.warning("DEV_OTP_CODE is set: OTPs are fixed to %s and nothing is sent. Remove before shipping.",
                DEV_OTP_CODE)
elif OTP_CHANNEL == "whatsapp" and not wa_ready(WA_OTP_CONTENT_SID):
    log.error("OTP_CHANNEL=whatsapp but not configured: need TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, WA_OTP_CONTENT_SID")
elif OTP_CHANNEL == "email" and not (SMTP_USER and SMTP_PASSWORD):
    log.error("OTP_CHANNEL=email but SMTP_USER / SMTP_PASSWORD are missing")
if not OTP_SECRET:
    log.error("Missing env: OTP_SECRET")
log.info("OTP channel: %s", OTP_CHANNEL)


class SignUpRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=128)
    full_name: str = Field(..., min_length=1, max_length=80)
    phone_number: str

    @field_validator("phone_number")
    @classmethod
    def _phone(cls, v: str) -> str:
        return normalize_in_phone(v)

    @field_validator("full_name")
    @classmethod
    def _name(cls, v: str) -> str:
        return v.strip()


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class RefreshRequest(BaseModel):
    refresh_token: str


class ForgotPasswordRequest(BaseModel):
    email: EmailStr


class ResetPasswordRequest(BaseModel):
    access_token: str
    new_password: str = Field(..., min_length=8, max_length=128)


class PhoneOTPRequest(BaseModel):
    phone_number: str

    @field_validator("phone_number")
    @classmethod
    def _phone(cls, v: str) -> str:
        return normalize_in_phone(v)


class PhoneOTPVerify(PhoneOTPRequest):
    otp: str = Field(..., min_length=4, max_length=8)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> User:
    try:
        res = await asyncio.to_thread(supabase_admin.auth.get_user, credentials.credentials)
    except Exception as e:
        log.info("token rejected: %s", e)
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or expired token")
    if not res or not res.user:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or expired token")
    return res.user


@router.post("/signup", status_code=status.HTTP_201_CREATED)
async def sign_up(payload: SignUpRequest):
    client = fresh_client()
    try:
        res = await db(lambda: client.auth.sign_up({
            "email": payload.email,
            "password": payload.password,
            "options": {"data": {"full_name": payload.full_name, "phone_number": payload.phone_number}},
        }))
    except Exception as e:
        log.error("signup failed: %r", e)
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Could not create account. Email may already be registered.")

    if not res.user:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Could not create account")

    # Supabase returns a fake user (empty identities) for an already-registered email
    if not res.user.identities:
        raise HTTPException(status.HTTP_409_CONFLICT, "Email already registered")

    # The email is stored on the profile so OTP sending never depends on the Auth admin API.
    profile_row = {
        "id": res.user.id,
        "full_name": payload.full_name,
        "phone_number": payload.phone_number,
        "email": str(payload.email),
    }
    try:
        try:
            await db(lambda: supabase_admin.table("profiles").upsert(profile_row).execute())
        except Exception as e:
            if "email" not in str(e).lower():
                raise
            log.warning("profiles.email column missing - run: alter table profiles add column if not exists email text;")
            profile_row.pop("email")
            await db(lambda: supabase_admin.table("profiles").upsert(profile_row).execute())
    except Exception as e:
        log.error("profile upsert failed for %s: %s", res.user.id, e)
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Could not create account")

    out = {"message": "User registered successfully", "user_id": res.user.id}
    if res.session:  # email confirmation disabled → log them straight in
        out.update(access_token=res.session.access_token, refresh_token=res.session.refresh_token,
                   token_type="bearer")
    return out


@router.post("/login")
async def login(payload: LoginRequest):
    client = fresh_client()
    try:
        res = await db(lambda: client.auth.sign_in_with_password(
            {"email": payload.email, "password": payload.password}))
    except Exception as e:
        log.info("login failed: %s", e)
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid email or password")

    if not res.session or not res.user:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid email or password")

    profile = {}
    try:
        p = await db(lambda: supabase_admin.table("profiles").select("full_name, phone_number, phone_verified")
                     .eq("id", res.user.id).maybe_single().execute())
        profile = (p.data if p else None) or {}
    except Exception as e:
        log.warning("profile fetch failed: %s", e)

    return {
        "access_token": res.session.access_token,
        "refresh_token": res.session.refresh_token,
        "expires_in": res.session.expires_in,
        "token_type": "bearer",
        "user_id": res.user.id,
        "full_name": profile.get("full_name"),
        "phone_number": profile.get("phone_number"),
        "phone_verified": bool(profile.get("phone_verified")),   # = "account verified" flag (OTP passed)
    }


@router.post("/refresh")
async def refresh(payload: RefreshRequest):
    client = fresh_client()
    try:
        res = await db(lambda: client.auth.refresh_session(payload.refresh_token))
    except Exception:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Refresh token invalid")
    if not res.session:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Refresh token invalid")
    return {
        "access_token": res.session.access_token,
        "refresh_token": res.session.refresh_token,
        "expires_in": res.session.expires_in,
        "token_type": "bearer",
    }


@router.post("/forgot-password")
async def forgot_password(payload: ForgotPasswordRequest):
    client = fresh_client()
    try:
        await db(lambda: client.auth.reset_password_for_email(
            payload.email, options={"redirect_to": RESET_REDIRECT_URL}))
    except Exception as e:
        log.info("reset email failed: %s", e)
    return {"message": "If an account exists with this email, a password reset link has been sent."}


@router.post("/reset-password")
async def reset_password(payload: ResetPasswordRequest):
    try:
        res = await asyncio.to_thread(supabase_admin.auth.get_user, payload.access_token)
        uid = res.user.id if res and res.user else None
    except Exception:
        uid = None
    if not uid:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Reset link invalid or expired")

    try:
        await db(lambda: supabase_admin.auth.admin.update_user_by_id(uid, {"password": payload.new_password}))
    except Exception as e:
        log.error("password update failed for %s: %s", uid, e)
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Password update failed")
    return {"message": "Password updated successfully."}


@router.get("/me")
async def read_current_user(
    current_user: User = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Depends(security),
):
    client = user_client(credentials.credentials)
    p = await db(lambda: client.table("profiles").select("full_name, phone_number, phone_verified, created_at")
                 .eq("id", current_user.id).maybe_single().execute())
    profile = (p.data if p else None) or {}
    return {
        "user_id": current_user.id,
        "email": current_user.email,
        "full_name": profile.get("full_name"),
        "phone_number": profile.get("phone_number"),
        "phone_verified": bool(profile.get("phone_verified")),
        "created_at": profile.get("created_at"),
    }


# ---------- OTP (email via SMTP, or WhatsApp via Twilio) ----------
# The client still calls /auth/phone/* with the phone number. We look the account up by phone and send the
# code to that account's email. A passed OTP sets profiles.phone_verified = true (the "account verified" flag).
def _hash_otp(phone: str, code: str) -> str:
    return hashlib.sha256(f"{phone}:{code}:{OTP_SECRET}".encode()).hexdigest()


def _parse_ts(s: str) -> datetime:
    """Postgres timestamps can have 1-5 fractional digits or 'Z'; fromisoformat on py<3.11 rejects those."""
    s = s.replace("Z", "+00:00")
    if "." in s:
        head, rest = s.split(".", 1)
        i = 0
        while i < len(rest) and rest[i].isdigit():
            i += 1
        s = f"{head}.{rest[:i][:6].ljust(6, '0')}{rest[i:]}"
    dt = datetime.fromisoformat(s)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _mask_email(email: str) -> str:
    name, _, domain = email.partition("@")
    return f"{name[:1]}***@{domain}" if domain else "***"


async def _profile_for_phone(phone: str) -> list[dict]:
    """Newest profile with this phone number. Reads profiles.email too; falls back if the column doesn't exist yet."""
    try:
        p = await db(lambda: supabase_admin.table("profiles").select("id, email")
                     .eq("phone_number", phone).order("created_at", desc=True).limit(1).execute())
    except Exception as e:
        if "email" not in str(e).lower():
            raise
        log.warning("profiles.email column missing - run: alter table profiles add column if not exists email text;")
        p = await db(lambda: supabase_admin.table("profiles").select("id")
                     .eq("phone_number", phone).order("created_at", desc=True).limit(1).execute())
    return (p.data if p else None) or []


async def _user_email(user_id: str) -> str | None:
    """Fallback only: Auth admin API lookup (may be rejected by new-style sb_secret keys)."""
    try:
        res = await db(lambda: supabase_admin.auth.admin.get_user_by_id(user_id))
        u = getattr(res, "user", None)
        return u.email if u and u.email else None
    except Exception as e:
        log.exception("otp: auth admin lookup failed for %s: %r", user_id, e)
        return None


def _smtp_send_sync(to_email: str, subject: str, body: str) -> None:
    msg = EmailMessage()
    msg["From"] = f"{SMTP_FROM_NAME} <{SMTP_FROM}>"
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.set_content(body)
    if SMTP_PORT == 465:
        with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, timeout=15) as s:
            s.login(SMTP_USER, SMTP_PASSWORD)
            s.send_message(msg)
    else:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=15) as s:
            s.starttls()
            s.login(SMTP_USER, SMTP_PASSWORD)
            s.send_message(msg)


async def _send_otp_email(email: str, code: str) -> tuple[bool, str]:
    if not (SMTP_USER and SMTP_PASSWORD and SMTP_FROM):
        return False, "SMTP_USER / SMTP_PASSWORD missing"
    body = (f"Your Suraksha verification code is {code}\n\n"
            f"It expires in {int(OTP_TTL.total_seconds() // 60)} minutes. "
            "Do not share it with anyone.\n\n"
            "If you didn't request this, you can ignore this email.")
    try:
        await asyncio.to_thread(_smtp_send_sync, email, f"{code} is your Suraksha verification code", body)
    except Exception as e:
        return False, f"smtp error: {e!r}"
    return True, "email sent"


async def _send_otp(phone: str, code: str, email: str | None) -> tuple[bool, str]:
    if DEV_OTP_CODE:
        log.warning("DEV OTP for %s: %s (nothing sent)", phone, code)
        return True, "dev-logged"
    if OTP_CHANNEL == "whatsapp":
        if not wa_ready(WA_OTP_CONTENT_SID):
            return False, "WhatsApp OTP not configured (TWILIO_* / WA_OTP_CONTENT_SID)"
        return await wa_send(phone, WA_OTP_CONTENT_SID, {"1": code})
    if not email:
        return False, "account has no email on file (profiles.email empty and auth admin lookup failed)"
    return await _send_otp_email(email, code)


@router.post("/phone/request-otp")
async def request_phone_otp(payload: PhoneOTPRequest):
    phone = payload.phone_number

    try:
        rows = await _profile_for_phone(phone)
    except Exception as e:
        log.exception("otp: profile lookup failed for ...%s: %r", phone[-4:], e)
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Could not look up phone number")
    if not rows:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Phone number not registered")
    user_id = rows[0]["id"]

    try:
        existing = await db(lambda: supabase_admin.table("otp_codes").select("last_sent_at")
                            .eq("user_id", user_id).limit(1).execute())
        ex = (existing.data if existing else None) or []
        if ex and ex[0].get("last_sent_at"):
            if datetime.now(timezone.utc) - _parse_ts(ex[0]["last_sent_at"]) < OTP_RESEND_COOLDOWN:
                raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "Wait a bit before requesting another OTP")
    except HTTPException:
        raise
    except Exception as e:
        log.exception("otp: cooldown check failed for %s: %r", user_id, e)   # don't block sending on this

    email = None
    if OTP_CHANNEL == "email" and not DEV_OTP_CODE:
        email = rows[0].get("email") or await _user_email(user_id)
        log.info("otp: email for %s is %s", user_id, "found" if email else "MISSING")

    code = DEV_OTP_CODE or f"{secrets.randbelow(1_000_000):06d}"
    now = datetime.now(timezone.utc)
    row = {
        "user_id": user_id,
        "phone_number": phone,
        "code_hash": _hash_otp(phone, code),
        "expires_at": (now + OTP_TTL).isoformat(),
        "attempts": 0,
        "last_sent_at": now.isoformat(),
    }
    try:
        await db(lambda: supabase_admin.table("otp_codes").upsert(row, on_conflict="user_id").execute())
    except Exception as e:
        log.exception("otp: row upsert failed for %s: %r", user_id, e)
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Could not start verification")

    ok, detail = await _send_otp(phone, code, email)
    if not ok:
        log.error("OTP send failed for %s via %s: %s", user_id, OTP_CHANNEL, detail)
        # don't leave a row behind, or the cooldown blocks the retry
        try:
            await db(lambda: supabase_admin.table("otp_codes").delete().eq("user_id", user_id).execute())
        except Exception as e:
            log.error("otp row cleanup failed for %s: %s", user_id, e)
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "Could not send OTP, try again")
    log.info("OTP sent for ...%s via %s: %s", phone[-4:], OTP_CHANNEL, detail)
    out = {"message": "OTP sent", "channel": OTP_CHANNEL}
    if email:
        out["sent_to"] = _mask_email(email)
    return out


@router.post("/phone/verify-otp")
async def verify_phone_otp(payload: PhoneOTPVerify):
    phone = payload.phone_number

    try:
        r = await db(lambda: supabase_admin.table("otp_codes").select("*")
                     .eq("phone_number", phone).order("last_sent_at", desc=True).limit(1).execute())
    except Exception as e:
        log.exception("otp verify lookup failed: %r", e)
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Could not verify right now")
    rows = (r.data if r else None) or []
    if not rows:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "No OTP pending for this number")
    row = rows[0]
    if row["attempts"] >= MAX_ATTEMPTS:
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "Too many attempts, request a new OTP")
    if _parse_ts(row["expires_at"]) < datetime.now(timezone.utc):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "OTP expired, request a new one")

    if not secrets.compare_digest(row["code_hash"], _hash_otp(phone, payload.otp)):
        await db(lambda: supabase_admin.table("otp_codes").update({"attempts": row["attempts"] + 1})
                 .eq("user_id", row["user_id"]).execute())
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Incorrect OTP")

    user_id = row["user_id"]
    await db(lambda: supabase_admin.table("profiles").update(
        {"phone_number": payload.phone_number, "phone_verified": True}).eq("id", user_id).execute())
    await db(lambda: supabase_admin.table("otp_codes").delete().eq("user_id", user_id).execute())
    return {"message": "Phone verified"}