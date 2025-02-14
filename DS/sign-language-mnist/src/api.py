import traceback
import uvicorn
import torch
import sys
import os

from fastapi import FastAPI, HTTPException, File, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from contextlib import asynccontextmanager
from io import BytesIO
from dotenv import load_dotenv

from ml.utils import ClassifierFactory, TClassifier, Converter
from common.logger import Logger
from dto.PredictionResponseDTO import PredictionResponseDTO

load_dotenv()

BASE_DIR = os.path.abspath(os.path.join(sys.argv[0], *[os.pardir] * 2))
MODEL_PATH = os.path.abspath(os.path.join(BASE_DIR, os.getenv("MODEL_PATH")))
LOG_DIR_PATH = os.path.abspath(os.path.join(BASE_DIR, os.getenv("LOG_DIR_PATH")))
LOG_FLNAME_API = os.getenv("LOG_FLNAME_API")

HOST = os.getenv("HOST")
PORT = int(os.getenv("PORT"))
API_PATH_PREDICTION = os.getenv("API_PATH_PREDICTION")

DEVICE = os.getenv("DEVICE")
DEBUG = os.getenv("DEBUG").lower() == "true"

logger = Logger(LOG_FLNAME_API, LOG_DIR_PATH)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Asynchronous context manager to handle the startup and shutdown of the FastAPI application.
    Loads the model weights during startup and sets the model in the app state.
    """
    logger.info("STARTUP")
    logger.info("Server data: " + str({
        "HOST": HOST, "PORT": PORT, "API_PATH_PREDICTION": API_PATH_PREDICTION
    }))
    logger.info("ML model data: " + str({
        "MODEL_PATH": MODEL_PATH, "DEVICE": DEVICE, "DEBUG": DEBUG
    }))
    logger.info("Loading model weights...")
    app.state.device = torch.device(DEVICE)
    app.state.model = ClassifierFactory.create(TClassifier.CNN, device=app.state.device)
    try:
        app.state.model.read_pth(MODEL_PATH)
    except Exception as e:
        logger.critical("Weights loading error: " + str({
            "error": e,
            "traceback": traceback.format_exc()
        }))
        app.state.model = None

    logger.info("Model weights loading completed")
    yield
    logger.info("SHUTDOWN")


app = FastAPI(lifespan=lifespan)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info({
        "IP": request.client.host,
        "Url path": request.url.path,
        "Query params": request.query_params,
    })
    response = await call_next(request)
    return response

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.error({
        "IP": request.client.host,
        "Url path": request.url.path,
        "Query params": request.query_params,
        "HTTPException": exc.detail
    })
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )


@app.post(f"/{API_PATH_PREDICTION}", summary="Character classification")
async def predict(
    request: Request, image: UploadFile = File(...)
) -> PredictionResponseDTO:
    """Predict the character from a gesture image."""
    if app.state.model is None:
        raise HTTPException(status_code=503, detail="Model is currently unavailable")

    img = await image.read()
    img_tensor = Converter.ImgToTensor(BytesIO(img))
    pred, probs_dict = app.state.model.predict(img_tensor)
    res = {"prediction": pred, "probabilities": probs_dict}
    logger.info({
            "IP": request.client.host,
            "Url path": request.url.path,
            "Query params": request.query_params,
            "Request":  res
        })
    return res


if __name__ == "__main__":
    """Runs the FastAPI application with Uvicorn."""
    uvicorn.run("api:app", host=HOST, port=PORT, reload=DEBUG)
