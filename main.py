import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import auth, sos, payload

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await sos.resume_pending_dispatches()  


app = FastAPI(title="Suraksha Core Engine", lifespan=lifespan)
app.include_router(auth.router)
app.include_router(sos.router)
app.include_router(payload.router)


@app.get("/health")
async def health():
    return {"ok": True}