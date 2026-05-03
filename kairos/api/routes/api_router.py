from __future__ import annotations
from pydantic import BaseModel
from fastapi import APIRouter, Request

router = APIRouter()


class InferRequest(BaseModel):
    instruction: str
    prompt: str
    answer: str
    kairos: str | None = None
    correct: bool | None = None


@router.post("/infer")
async def infer(req: InferRequest, request: Request):
    result = await request.app.state.core_client.infer(req.model_dump())

    return result
