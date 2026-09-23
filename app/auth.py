import asyncio
import logging
import os

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, EmailStr, Field, field_validator

from app.db import User, db, fresh_client, normalize_in_phone, supabase_admin, user_client

log = logging.getLogger("suraksha.auth")
security = HTTPBearer()
router = APIRouter(prefix="/auth", tags=["Authentication"])

RESET_REDIRECT_URL = os.getenv("RESET_REDIRECT_URL", "https://suraksha-app.com/reset-password")


# ---------- models ----------
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


# ---------- dependency ----------
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


# ---------- routes ----------
@router.post("/signup", status_code=status.HTTP_201_CREATED)
async def sign_up(payload: SignUpRequest):
    client = fresh_client()  # never sign up on the shared admin client
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
    """Safety app must never be logged out mid-emergency — client refreshes before expiry."""
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
        log.info("reset email failed: %s", e)  # swallow: don't reveal whether account exists
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
