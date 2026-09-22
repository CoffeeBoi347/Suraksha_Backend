from fastapi import APIRouter, Depends, BackgroundTasks, status, HTTPException
from pydantic import BaseModel, Field
from supabase import User
from app.auth import get_current_user, supabase
from dotenv import load_dotenv
import os
import httpx

load_dotenv()

fast2sms_api_key = os.getenv("FAST2_SMS_API_KEY")
fast2sms_url = os.getenv("FAST2_SMS_URL")

router = APIRouter(prefix="/sos", tags=["SOS Emergency"])

class SOSEmergencyRequest(BaseModel):
    latitude: float
    longitude: float
    accuracy_m: float = 5.0
    speed_mps: float = 0.0
    battery_percentage: int = Field(..., ge=0, le=100)
    snapshot_url: str | None = None

class ContactRequest(BaseModel):
    name: str
    phone_number: str

async def send_fast2sms_alert(phone_numbers: list[str], victim_name:str, map_link: str):
    """
    Background worker task to dispatch SMS via Fast2SMS quick route
    """

    if not fast2sms_api_key:
        print("FAST2SMS API KEY is not initialized in environment variables.")
        return

    if not phone_numbers:
        print("No emergency contacts registered for this user.")
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
                print(f"SMS dispatched to clean numbers. Request ID: {res_data.get('request_id')}")
            else:
                print(f"Provider Error: {res_data.get('message')}")

        except Exception as e:
            print(f"SMS failed to reach endpoint. Reason: {str(e)}")

@router.post("/trigger", status_code=status.HTTP_200_OK)
async def trigger_sos(payload: SOSEmergencyRequest, background_tasks: BackgroundTasks, current_user: User = Depends(get_current_user)):
    user_id = current_user.id
    user_metadata = current_user.user_metadata or {}
    user_name = user_metadata.get("full_name", "Suraksha User")

    maps_link = f"https://maps.google.com/?q={payload.latitude},{payload.longitude}" 
    incident_data = {
        "user_id": user_id,
        "latitude": payload.latitude,
        "longitude": payload.longitude,
        "accuracy_m": payload.accuracy_m,
        "speed_mps": payload.speed_mps,
        "battery_percentage": payload.battery_percentage,
        "snapshot_url": payload.snapshot_url,
        "status": "ACTIVE"
    }

    try:
        db_response = supabase.table("sos_incidents").insert(incident_data).execute()
        incident_id = db_response.data[0]["id"] if db_response.data else None
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to log emergency incident: {str(e)}"
        )

    try:
        contacts_response = (
            supabase.table("emergency_contacts")
            .select("phone_number")
            .eq("user_id", user_id)
            .execute()
        )
        phone_numbers = [c["phone_number"] for c in contacts_response.data] if contacts_response.data else []
    except Exception as e:
        phone_numbers = []
        print(f"[Warning] Failed to fetch emergency contacts: {str(e)}")

    background_tasks.add_task(
        send_fast2sms_alert,
        phone_numbers=phone_numbers,
        victim_name=user_name,
        map_link=maps_link
    )

    return {
        "status": "SOS_DISPATCHED",
        "incident_id": incident_id,
        "maps_link": maps_link,
        "contacts_notified": len(phone_numbers)
    }

# Contacts

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