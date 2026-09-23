"""Shared Supabase clients + helpers."""
import asyncio
import os
import re

from dotenv import load_dotenv
from supabase import Client, create_client

try:  # location of User moved across supabase-py versions
    from supabase_auth.types import User  # noqa: F401
except ImportError:
    from gotrue.types import User  # noqa: F401

load_dotenv()

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_KEY = os.environ["SUPABASE_SERVICE_KEY"]  # service_role — server only, never ship to client
SUPABASE_ANON_KEY = os.environ["SUPABASE_ANON_KEY"]        # anon — used for per-request user sessions

# Admin client: NEVER call sign_in / sign_up / set_session on this.
# Doing so stores a user session on a shared object and leaks it across requests.
supabase_admin: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)


def fresh_client() -> Client:
    """Throwaway client for anything that creates/holds a user session."""
    return create_client(SUPABASE_URL, SUPABASE_ANON_KEY)


def user_client(access_token: str) -> Client:
    """Client whose DB queries run as the user (RLS applies)."""
    client = fresh_client()
    client.postgrest.auth(access_token)
    return client


async def db(fn):
    """supabase-py is sync — run calls off the event loop. Usage: await db(lambda: ...execute())"""
    return await asyncio.to_thread(fn)


def normalize_in_phone(raw: str) -> str:
    """Normalize an Indian mobile number to +91XXXXXXXXXX or raise ValueError."""
    digits = re.sub(r"\D", "", raw or "")
    if len(digits) == 12 and digits.startswith("91"):
        digits = digits[2:]
    elif len(digits) == 11 and digits.startswith("0"):
        digits = digits[1:]
    if len(digits) != 10 or digits[0] not in "6789":
        raise ValueError("Enter a valid 10-digit Indian mobile number")
    return "+91" + digits
