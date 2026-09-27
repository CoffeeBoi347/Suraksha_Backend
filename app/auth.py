import asyncio
import hashlib
import logging
import os
import secrets
from datetime import datetime, timedelta, timezone

import httpx
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, EmailStr, Field, field_validator

from app.db import User, db, fresh_client, normalize_in_phone, supabase_admin, user_client

log = logging.getLogger("suraksha.auth")
security = HTTPBearer()
router = APIRouter(prefix="/auth", tags=["Authentication"])

RESET_REDIRECT_URL = os.getenv("RESET_REDIRECT_URL", "https://suraksha-app.com/reset-password")

FAST2SMS_API_KEY = os.getenv("FAST2SMS_API_KEY")
FAST2SMS_URL = os.getenv("FAST2SMS_URL")
OTP_SECRET = os.getenv("OTP_SECRET")
OTP_TTL = timedelta(minutes=5)
OTP_RESEND_COOLDOWN = timedelta(seconds=60)
MAX_ATTEMPTS = 5


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
        log.info("signup failed: %s", e)
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Could not create account. Email may already be registered.")

    if not res.user:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Could not create account")

    try:
        await db(lambda: supabase_admin.table("profiles").upsert({
            "id": res.user.id,
            "full_name": payload.full_name,
            "phone_number": payload.phone_number,
        }).execute())
    except Exception as e:
        log.error("profile upsert failed for %s: %s", res.user.id, e)

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
        p = await db(lambda: supabase_admin.table("profiles").select("full_name, phone_number")
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
    p = await db(lambda: client.table("profiles").select("full_name, phone_number, created_at")
                 .eq("id", current_user.id).maybe_single().execute())
    profile = (p.data if p else None) or {}
    return {
        "user_id": current_user.id,
        "email": current_user.email,
        "full_name": profile.get("full_name"),
        "phone_number": profile.get("phone_number"),
        "created_at": profile.get("created_at"),
    }


def _hash_otp(phone: str, code: str) -> str:
    return hashlib.sha256(f"{phone}:{code}:{OTP_SECRET}".encode()).hexdigest()


async def _send_otp_sms(phone: str, code: str) -> tuple[bool, str]:
    if not FAST2SMS_API_KEY:
        return False, "FAST2SMS_API_KEY missing"
    digits = "".join(c for c in phone if c.isdigit())[-10:]
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.post(FAST2SMS_URL,
            headers={"authorization": FAST2SMS_API_KEY},
            data={"route": "otp", "variables_values": code, "numbers": digits})
    try:
        body = r.json()
    except ValueError:
        return False, f"HTTP {r.status_code}: {r.text[:200]}"
    return (True, "sent") if body.get("return") else (False, str(body.get("message")))


@router.post("/phone/request-otp")
async def request_phone_otp(payload: PhoneOTPRequest, current_user: User = Depends(get_current_user)):
    phone = payload.phone_number
    existing = await db(lambda: supabase_admin.table("otp_codes").select("last_sent_at")
                        .eq("user_id", current_user.id).maybe_single().execute())
    if existing and existing.data:
        last = datetime.fromisoformat(existing.data["last_sent_at"])
        if datetime.now(timezone.utc) - last < OTP_RESEND_COOLDOWN:
            raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "Wait a bit before requesting another OTP")

    code = f"{secrets.randbelow(1_000_000):06d}"
    row = {
        "user_id": current_user.id,
        "phone_number": phone,
        "code_hash": _hash_otp(phone, code),
        "expires_at": (datetime.now(timezone.utc) + OTP_TTL).isoformat(),
        "attempts": 0,
        "last_sent_at": datetime.now(timezone.utc).isoformat(),
    }
    await db(lambda: supabase_admin.table("otp_codes").upsert(row).execute())

    ok, detail = await _send_otp_sms(phone, code)
    if not ok:
        log.error("OTP send failed for %s: %s", current_user.id, detail)
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "Could not send OTP, try again")
    return {"message": "OTP sent"}


@router.post("/phone/verify-otp")
async def verify_phone_otp(payload: PhoneOTPVerify, current_user: User = Depends(get_current_user)):
    r = await db(lambda: supabase_admin.table("otp_codes").select("*")
                 .eq("user_id", current_user.id).maybe_single().execute())
    row = r.data if r else None
    if not row or row["phone_number"] != payload.phone_number:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "No OTP pending for this number")
    if row["attempts"] >= MAX_ATTEMPTS:
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "Too many attempts, request a new OTP")
    if datetime.fromisoformat(row["expires_at"]) < datetime.now(timezone.utc):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "OTP expired, request a new one")

    if not secrets.compare_digest(row["code_hash"], _hash_otp(payload.phone_number, payload.otp)):
        await db(lambda: supabase_admin.table("otp_codes").update({"attempts": row["attempts"] + 1})
                 .eq("user_id", current_user.id).execute())
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Incorrect OTP")

    await db(lambda: supabase_admin.table("profiles").update(
        {"phone_number": payload.phone_number, "phone_verified": True}).eq("id", current_user.id).execute())
    await db(lambda: supabase_admin.table("otp_codes").delete().eq("user_id", current_user.id).execute())
    return {"message": "Phone verified"}