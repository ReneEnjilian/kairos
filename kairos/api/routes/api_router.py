from __future__ import annotations
from pydantic import BaseModel
from fastapi import APIRouter, Request
import time
router = APIRouter()


class InferRequest(BaseModel):
    instruction: str
    prompt: str
    answer: str
    kairos: str | None = None
    correct: bool | None = None
    arrival_timestamp: float | None = None
    infer_latency_ms: float | None = None


@router.post("/infer")
async def infer(req: InferRequest, request: Request):
    payload = req.model_dump()

    payload["arrival_timestamp"] = time.time()

    result = await request.app.state.core_client.infer(payload)

    return result
