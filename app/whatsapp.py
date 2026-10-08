"""Twilio WhatsApp via Content API templates (OTP + SOS + SAFE). Async httpx, no Twilio SDK.

Business-initiated WhatsApp messages MUST use an approved template (ContentSid HX...).
Twilio returns 201 when it accepts the message, not when it delivers it. "Not on WhatsApp" and similar
failures arrive later, asynchronously (check Twilio Console > Monitor > Logs > Messaging).
"""
import json
import logging
import os
from pathlib import Path

import httpx
from dotenv import load_dotenv

# .env sits in the project root (parent of app/). Must load BEFORE the os.getenv calls below.
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

log = logging.getLogger("suraksha.whatsapp")

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_WA_FROM = os.getenv("TWILIO_WA_FROM", "whatsapp:+918792598899")
TWILIO_MESSAGING_SERVICE_SID = os.getenv("TWILIO_MESSAGING_SERVICE_SID", "")  # optional MG..., overrides From

WA_OTP_CONTENT_SID = os.getenv("WA_OTP_CONTENT_SID", "")    # only used if OTP_CHANNEL=whatsapp
WA_SOS_CONTENT_SID = os.getenv("WA_SOS_CONTENT_SID", "")    # Utility: {{1}} name, {{2}} coords, {{3}} maps link
WA_SAFE_CONTENT_SID = os.getenv("WA_SAFE_CONTENT_SID", "")  # Utility: {{1}} name

_API = "https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json"
_DEAD = {"failed", "undelivered", "canceled"}

if not (TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN):
    log.warning("Twilio WhatsApp disabled: TWILIO_ACCOUNT_SID / TWILIO_AUTH_TOKEN missing")


def wa_ready(content_sid: str | None) -> bool:
    return bool(TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN and content_sid)


def _to_wa(num: str) -> str:
    d = "".join(c for c in num if c.isdigit())
    return f"whatsapp:+91{d[-10:]}"


def _clean(v) -> str:
    # Meta rejects template variables with newlines, tabs or 4+ consecutive spaces
    return " ".join(str(v).split())[:200] or "-"


async def wa_send(to: str, content_sid: str | None, variables: dict) -> tuple[bool, str]:
    """-> (accepted_by_twilio, detail). Drops straight into _with_retry."""
    if not wa_ready(content_sid):
        return False, "whatsapp not configured"
    data = {
        "To": _to_wa(to),
        "ContentSid": content_sid,
        "ContentVariables": json.dumps({str(k): _clean(v) for k, v in variables.items()}),
    }
    if TWILIO_MESSAGING_SERVICE_SID:
        data["MessagingServiceSid"] = TWILIO_MESSAGING_SERVICE_SID
    else:
        data["From"] = TWILIO_WA_FROM
    try:
        async with httpx.AsyncClient(timeout=10.0, auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)) as c:
            r = await c.post(_API.format(sid=TWILIO_ACCOUNT_SID), data=data)
    except httpx.HTTPError as e:
        return False, f"network error: {e}"
    try:
        body = r.json()
    except ValueError:
        return False, f"HTTP {r.status_code}: {r.text[:200]}"
    if r.status_code in (200, 201) and body.get("status") not in _DEAD:
        return True, f"sid={body.get('sid')} status={body.get('status')}"
    return False, f"HTTP {r.status_code}: code={body.get('code')} {body.get('message')}"