import os
from supabase import Client, create_client  
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

try:
    response = supabase.table("profiles").select("*").limit(1).execute()
    print("Successfully connected to Supabase")
except Exception as e:
    print(f"Connection failed. Reason: {str(e)}")