from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from app.auth import get_current_user, supabase

router = APIRouter(prefix="/sos", tags=["SOS Emergency"])

class EmergencyContact(BaseModel):
    contact_name: str
    phone_number: str

@router.post("/contacts")
async def add_emergency_contact(contact: EmergencyContact, user = Depends(get_current_user)):
    res = supabase.table("emergency_contacts").insert({
        "user_id": user.id,
        "contact_name": contact.contact_name,
        "phone_number": contact.phone_number
    }).execute()

    return { "status" : "contact_added", "data" : res.data }

@router.post("/trigger")
async def trigger_sos(lat: float, lng: float, user=Depends(get_current_user)):
    supabase.table("incident_logs").insert({
        "user_id": user.id,
        "latitude": lat,
        "longitude": lng,
        "resolved": False
    }).execute()

    return {"status": "SOS_DISPATCHED", "message": "Emergency contacts notifed."}