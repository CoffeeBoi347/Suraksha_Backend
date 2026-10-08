import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import auth, sos, vision

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await sos.resume_pending_dispatches()  # recover SOS dispatches interrupted by a crash/redeploy
    yield


app = FastAPI(title="Suraksha Core Engine", lifespan=lifespan)
app.include_router(auth.router)
app.include_router(sos.router)
app.include_router(vision.router)


@app.get("/health")
async def health():
    return {"ok": True}