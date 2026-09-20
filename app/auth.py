import os
from dotenv import load_dotenv
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, EmailStr, Field
from supabase import Client, create_client

load_dotenv()

SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
security = HTTPBearer()

router = APIRouter(prefix="/auth", tags=["Authentication"])

class SignUpRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=6)
    full_name: str
    phone_number: str

class LoginRequest(BaseModel):
    email: EmailStr
    password: str 

class ForgotPasswordRequest(BaseModel):
    email: EmailStr

class ResetPasswordRequest(BaseModel):
    access_token: str
    new_password: str = Field(..., min_length=6)

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    try:
        user_response = supabase.auth.get_user(token)
        if not user_response or not user_response.user:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid Token")
        
        return user_response.user

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Authentication Failed. Reason: {str(e)}"
        )


@router.post("/signup", status_code=status.HTTP_201_CREATED)
async def sign_up(payload: SignUpRequest):
    try:
        response = supabase.auth.sign_up(
            {
                "email": payload.email,
                "password": payload.password,
                "options": {
                    "data": {
                        "full_name": payload.full_name,
                        "phone_number": payload.phone_number
                    }
                }
            }
        )

        return{
            "message": "User registered successfully",
            "user_id": response.user.id if response.user else None,
        }

    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

@router.post("/login")
async def login(payload: LoginRequest):
    try:
        response = supabase.auth.sign_in_with_password({
            "email": payload.email,
            "password": payload.password
        })

        return{
            "access_token": response.session.access_token,
            "refresh_token": response.session.refresh_token,
            "token_type": "bearer",
            "user_id": response.user.id
        }

    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

@router.post("/forgot-password")
async def forgot_password(payload: ForgotPasswordRequest):
    try:
        supabase.auth.reset_password_for_email(
            payload.email,
            options={"redirect_to": "https://suraksha-app.com/reset-password"},
        )
        return {
            "message": "If an account exists with this email, a password reset link has been sent."
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)
        )

@router.post("/reset-password")
async def reset_password(payload: ResetPasswordRequest):
    try:
        supabase.auth.api.update_user_by_jwt(
            payload.access_token, {"password": payload.new_password}
        )
        return {"message": "Password updated successfully."}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)
        )

router.get("/me")
async def read_current_user(current_user=Depends(get_current_user)):
    return {"user_id": current_user.id, "email": current_user.email}