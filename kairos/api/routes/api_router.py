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


@router.post("/infer")
async def infer(req: InferRequest, request: Request):
    t0 = time.time()
    result = await request.app.state.core_client.infer(req.model_dump())
    t1 = time.time()
    print(f"time for complete round: {(t1 - t0)*1000}")
    return result
