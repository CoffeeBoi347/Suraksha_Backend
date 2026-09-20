import uvicorn
from fastapi import FastAPI
from app.auth import router as auth_router
from app.payload import app as payload_app, main_suraksha
from app.sos import router as sos_router

app = FastAPI(itle="Suraksha Core Engine")

app.include_router(auth_router)
app.add_api_websocket_route("/prod/main", main_suraksha)
app.include_router(sos_router)

@app.get("/")
async def root():
    return {"status": "online", "system": "Suraksha Backend API"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)