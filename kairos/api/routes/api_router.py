from __future__ import annotations

from fastapi import APIRouter, Request
router = APIRouter()


@router.post("/infer")
async def infer(request: Request):
    print("[API] doing API work ...")

    result = await request.app.state.core_client.infer("hello from api")

    return {"message": result}
