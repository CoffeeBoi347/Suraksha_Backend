import os
import re
import time
from collections import deque
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from app.db import db, supabase_admin

router = APIRouter(prefix="/waitlist", tags=["Waitlist"])

RATE_MAX = int(os.getenv("WAITLIST_RATE_MAX", "5"))          # signups per IP...
RATE_WINDOW_S = int(os.getenv("WAITLIST_RATE_WINDOW_S", "600"))  # ...per 10 min
_hits: dict[str, deque] = {}
_count_cache: tuple[float, int] = (0.0, 0)

UNSAFE = {"commute", "night_walk", "cab_auto", "late_work", "hostel_pg", "crowds"}
CITIES = {"Bengaluru", "Mumbai", "Delhi NCR", "Hyderabad", "Pune", "Chennai", "Kolkata", "Other"}
EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]{2,}$")


class Signup(BaseModel):
    name: str = Field(min_length=2, max_length=80)
    email: str = Field(max_length=120)
    phone: str | None = Field(default=None, max_length=16)
    city: str
    persona: Literal["student", "working", "parent", "institution"]
    institution: str | None = Field(default=None, max_length=120)
    unsafe_when: list[str] = Field(default_factory=list, max_length=6)
    phone_model: str | None = Field(default=None, max_length=60)
    beta: bool = False
    consent: bool
    source: str | None = Field(default=None, max_length=40)
    referrer: str | None = Field(default=None, max_length=200)
    website: str | None = None                                   # honeypot

    @field_validator("name", "institution", "phone_model", "source")
    @classmethod
    def _strip(cls, v):
        return v.strip() if isinstance(v, str) else v

    @field_validator("email")
    @classmethod
    def _email(cls, v):
        v = v.strip().lower()
        if not EMAIL_RE.match(v):
            raise ValueError("invalid email")
        return v

    @field_validator("phone")
    @classmethod
    def _phone(cls, v):
        if not v:
            return None
        d = re.sub(r"\D", "", v)
        if len(d) == 12 and d.startswith("91"):
            d = d[2:]
        if len(d) == 11 and d.startswith("0"):
            d = d[1:]
        if not re.fullmatch(r"[6-9]\d{9}", d):
            raise ValueError("invalid phone")
        return "+91" + d

    @field_validator("city")
    @classmethod
    def _city(cls, v):
        if v not in CITIES:
            raise ValueError("invalid city")
        return v

    @field_validator("unsafe_when")
    @classmethod
    def _unsafe(cls, v):
        return [x for x in dict.fromkeys(v) if x in UNSAFE]


def _client_ip(req: Request) -> str:
    fwd = req.headers.get("x-forwarded-for", "")
    return fwd.split(",")[0].strip() if fwd else (req.client.host if req.client else "?")


def _rate_ok(ip: str) -> bool:
    now = time.time()
    q = _hits.setdefault(ip, deque())
    while q and now - q[0] > RATE_WINDOW_S:
        q.popleft()
    if len(q) >= RATE_MAX:
        return False
    q.append(now)
    if len(_hits) > 5000:                                        # stop the dict growing forever
        for k in [k for k, v in _hits.items() if not v]:
            _hits.pop(k, None)
    return True


@router.post("")
async def join(body: Signup, request: Request):
    if body.website:                                             # bot filled the hidden field
        return {"status": "ok"}
    if not body.consent:
        raise HTTPException(400, "consent required")
    if body.persona == "institution" and not body.institution:
        raise HTTPException(400, "institution name required")
    if not _rate_ok(_client_ip(request)):
        raise HTTPException(429, "too many signups, try again later")

    row = body.model_dump(exclude={"website"})
    if body.persona != "institution":
        row["institution"] = None
    try:
        await db(lambda: supabase_admin.table("waitlist").insert(row).execute())
    except Exception as e:
        msg = str(e)
        if "23505" in msg or "duplicate" in msg.lower():
            return {"status": "already"}
        raise HTTPException(500, "could not save")
    return {"status": "ok"}


@router.get("/count")
async def count():
    global _count_cache
    now = time.time()
    if now - _count_cache[0] < 60:
        return {"count": _count_cache[1]}
    try:
        res = await db(lambda: supabase_admin.table("waitlist").select("id", count="exact").limit(1).execute())
        n = int(getattr(res, "count", 0) or 0)
    except Exception:
        n = _count_cache[1]
    _count_cache = (now, n)
    return {"count": n}