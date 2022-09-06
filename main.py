import uvicorn
from fastapi import FastAPI
from loguru import logger
from mlflow.tracking import MlflowClient

from routers.system_router import system_router


app = FastAPI()
app.include_router(system_router)
client = MlflowClient()


@app.on_event("startup")
async def startup_event() -> None:
    logger.info("Starting app")


@app.on_event("shutdown")
async def shutdown_event() -> None:
    logger.info("Shutting down...")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0")
