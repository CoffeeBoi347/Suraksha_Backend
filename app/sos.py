import os
import httpx
from fastapi import APIRouter, Depends, BackgroundTasks, status, HTTPException
from pydantic import BaseModel, Field
from supabase import User
from app.auth import get_current_user, supabase
from dotenv import load_dotenv

load_dotenv()

fast2sms_api_key = os.getenv("FAST2_SMS_API_KEY")
fast2sms_url = os.getenv("FAST2_SMS_URL", "https://www.fast2sms.com/dev/bulkV2")

exotel_sid = os.getenv("EXOTEL_ACCOUNT_SID")
exotel_caller_id = os.getenv("EXOTEL_CALLER_ID") 
exotel_app_id = os.getenv("EXOTEL_APP_ID")      
exotel_api_key = os.getenv("EXOTEL_API_KEY")
exotel_api_tokens = os.getenv("EXOTEL_API_TOKEN")

router = APIRouter(prefix="/sos", tags=["SOS Emergency Engine"])

class SOSEmergencyRequest(BaseModel):
    latitude: float
    longitude: float
    state_code: str = "NATIONAL"
    district: str = "All"
    accuracy_m: float = 5.0
    speed_mps: float = 0.0
    battery_percentage: int = Field(..., ge=0, le=100)
    snapshot_url: str | None = None
    threat_level: str = "CRITICAL"

class ContactRequest(BaseModel):
    name: str
    phone_number: str


async def send_fast2sms_alert(phone_numbers: list[str], victim_name: str, map_link: str):
    """
    Background worker task to dispatch SMS via Fast2SMS quick route.
    """
    if not fast2sms_api_key:
        print("[SMS Error] FAST2SMS API KEY is not initialized in environment variables.")
        return

    if not phone_numbers:
        print("[SMS Notice] No emergency contacts provided for dispatch.")
        return

    clean_numbers = ",".join([p.replace("+91", "").strip() for p in phone_numbers])
    message_text = (
        f"🚨 SURAKSHA EMERGENCY ALERT! 🚨\n"
        f"{victim_name} has triggered an SOS!\n"
        f"Live Map: {map_link}\n"
        f"Please check in immediately."
    )

    headers = {
        "authorization": fast2sms_api_key,
        "accept": "application/json"
    }

    payload = {
        "route": "q",
        "message": message_text,
        "numbers": clean_numbers
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(fast2sms_url, headers=headers, data=payload, timeout=10.0)
            res_data = response.json()

            if res_data.get("return"):
                print(f"[SMS Success] SMS dispatched to {clean_numbers}. Request ID: {res_data.get('request_id')}")
            else:
                print(f"[SMS Provider Error] {res_data.get('message')}")

        except Exception as e:
            print(f"[SMS Exception] Failed to reach endpoint: {str(e)}")


async def trigger_exotel_ivr_call(target_phone: str, victim_name: str, map_link: str):
    """
    Triggers an automated voice call to official helpline numbers (112, 1091, 1090) via Exotel.
    """
    if not exotel_sid or not exotel_api_key or not exotel_api_tokens:
        print("[Exotel Error] Exotel credentials missing in environment.")
        return

    url = f"https://api.exotel.com/v1/Accounts/{exotel_sid}/Calls/connect.json"
    formatted_target = target_phone if target_phone.startswith("+") else f"0{target_phone.lstrip('0')}"

    payload = {
        "From": formatted_target,
        "CallerId": exotel_caller_id,
        "Url": f"http://my.exotel.in/exoml/start/{exotel_app_id}",
        "CallType": "trans",
        "CustomField": f"User: {victim_name} | Map: {map_link}"
    }

    auth = (exotel_api_key, exotel_api_tokens)

    async with httpx.AsyncClient(auth=auth) as client:
        try:
            res = await client.post(url, data=payload, timeout=10.0)
            if res.status_code == 200:
                print(f"[Exotel Success] Call queued to {target_phone} successfully.")
            else:
                print(f"[Exotel Error] Status Code: {res.status_code} | Text: {res.text}")

        except Exception as e:
            print(f"[Exotel Exception] Failed to place call to {target_phone}: {str(e)}")


@router.get("/tts-message")
async def get_exotel_tts_message(CustomField: str = ""):
    """
    Called dynamically by Exotel Passthru Applet inside Flow 1346134 when the official desk picks up.
    Returns plain text that Exotel converts to Text-To-Speech (TTS).
    """
    # Sanitize CustomField string for clean voice reading
    info_text = CustomField.replace("|", " comma ").replace("https://", "") if CustomField else "Location unknown"

    tts_content = (
        f"Critical Emergency alert from Suraksha Safety System. "
        f"A user requires immediate assistance. "
        f"Details: {info_text}. "
        f"An official dispatch notification with live location coordinates has also been logged."
    )
    return {"select_data": tts_content}

@router.post("/trigger", status_code=status.HTTP_200_OK)
async def trigger_sos(
    payload: SOSEmergencyRequest, 
    background_tasks: BackgroundTasks, 
    current_user: User = Depends(get_current_user)
):
    user_id = current_user.id
    user_metadata = current_user.user_metadata or {}
    user_name = user_metadata.get("full_name", "Suraksha User")

    maps_link = f"https://maps.google.com/?q={payload.latitude},{payload.longitude}"

    try:
        contacts_response = (
            supabase.table("emergency_contacts")
            .select("phone_number")
            .eq("user_id", user_id)
            .execute()
        )
        personal_numbers = [c["phone_number"] for c in contacts_response.data] if contacts_response.data else []
    except Exception as e:
        personal_numbers = []
        print(f"[Warning] Failed to fetch emergency contacts: {str(e)}")

    try:
        desk_query = (
            supabase.table("safety_desks")
            .select("*")
            .eq("state_code", payload.state_code.upper())
            .eq("district", payload.district)
            .eq("is_active", True)
            .execute()
        )
        desk = desk_query.data[0] if desk_query.data else None

        if not desk:
            state_query = (
                supabase.table("safety_desks")
                .select("*")
                .eq("state_code", payload.state_code.upper())
                .eq("is_active", True)
                .execute()
            )
            desk = state_query.data[0] if state_query.data else None

        if not desk:
            national_query = (
                supabase.table("safety_desks")
                .select("*")
                .eq("state_code", "NATIONAL")
                .execute()
            )
            desk = national_query.data[0] if national_query.data else {
                "organization_name": "Pan-India Emergency Response (112)",
                "phone_number": "112",
                "can_receive_sms": False
            }

    except Exception as e:
        print(f"[Warning] Safety desk lookup failed: {str(e)}")
        desk = {"organization_name": "Pan-India Emergency Response (112)", "phone_number": "112", "can_receive_sms": False}

    target_desk_phone = desk["phone_number"]
    can_desk_sms = desk.get("can_receive_sms", False)

    if personal_numbers:
        background_tasks.add_task(
            send_fast2sms_alert,
            phone_numbers=personal_numbers,
            victim_name=user_name,
            map_link=maps_link
        )

    if can_desk_sms:
        background_tasks.add_task(
            send_fast2sms_alert,
            phone_numbers=[target_desk_phone],
            victim_name=f"{user_name} [OFFICIAL DISPATCH]",
            map_link=maps_link
        )
    else:
        background_tasks.add_task(
            trigger_exotel_ivr_call,
            target_phone=target_desk_phone,
            victim_name=user_name,
            map_link=maps_link
        )

    incident_data = {
        "user_id": user_id,
        "latitude": payload.latitude,
        "longitude": payload.longitude,
        "accuracy_m": payload.accuracy_m,
        "speed_mps": payload.speed_mps,
        "battery_percentage": payload.battery_percentage,
        "snapshot_url": payload.snapshot_url,
        "dispatched_to_desk": desk.get("organization_name"),
        "dispatched_phone": target_desk_phone,
        "status": "ACTIVE"
    }

    try:
        db_response = supabase.table("sos_incidents").insert(incident_data).execute()
        incident_id = db_response.data[0]["id"] if db_response.data else None
    except Exception as e:
        print(f"[Error] Log incident failed: {str(e)}")
        incident_id = None

    return {
        "status": "SOS_DISPATCHED",
        "incident_id": incident_id,
        "maps_link": maps_link,
        "contacts_notified": len(personal_numbers),
        "official_desk_notified": desk.get("organization_name"),
        "official_dispatch_type": "FAST2SMS" if can_desk_sms else "EXOTEL_IVR_CALL"
    }

@router.post("/contacts")
async def add_contact(payload: ContactRequest, current_user: User = Depends(get_current_user)):
    data = {
        "user_id": current_user.id,
        "name": payload.name,
        "phone_number": payload.phone_number
    }
    res = supabase.table("emergency_contacts").insert(data).execute()
    return {"message": "Contact added successfully", "data": res.data}


@router.get("/contacts")
async def get_contacts(current_user: User = Depends(get_current_user)):
    res = (
        supabase.table("emergency_contacts")
        .select("*")
        .eq("user_id", current_user.id)
        .execute()
    )
    return {"contacts": res.data if res.data else []}


@router.delete("/contacts/{contact_id}")
async def delete_contact(contact_id: str, current_user: User = Depends(get_current_user)):
    supabase.table("emergency_contacts").delete().eq("id", contact_id).eq("user_id", current_user.id).execute()
    return {"message": "Contact deleted successfully"}