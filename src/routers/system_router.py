from fastapi import APIRouter

system_router = APIRouter(tags=["System"])


@system_router.get("/healthz")
async def health() -> dict[str, bool]:
    return {"ok": True}


@system_router.get("/readyz")
async def ready() -> dict[str, bool]:
    return {"ok": True}
