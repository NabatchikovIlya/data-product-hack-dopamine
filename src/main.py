import uvicorn
from fastapi import FastAPI
from loguru import logger

from data import PreprocessingPipe
from features import FeatureBuilder
from routers.system_router import system_router
from services.dofamine_service import DofamineService

app = FastAPI()
app.include_router(system_router)
preprocessing_pipe = PreprocessingPipe()
feature_builder = FeatureBuilder()
dofamine_service = DofamineService(
    preprocessing_pipe=preprocessing_pipe,
    feature_builder=feature_builder,
    model_path='../models/pipeline.pickle',
)


@app.on_event("startup")
async def startup_event() -> None:
    logger.info("Starting app")


@app.post("/get_test_results")
def get_test_results(data: dict[str, str]):
    results = dofamine_service.get_test_results(data=data)
    return results


@app.on_event("shutdown")
async def shutdown_event() -> None:
    logger.info("Shutting down...")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0")
