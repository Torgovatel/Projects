import string
import traceback
from pydantic import BaseModel, condecimal, constr
import uvicorn
import torch
import os
import sys
import numpy as np

from fastapi import FastAPI, HTTPException, File, UploadFile
from contextlib import asynccontextmanager
from termcolor import colored
from io import BytesIO

from ml.utils import ClassifierFactory, TClassifier, Converter
from dto.PredictionResponseDTO import PredictionResponseDTO

base_dir = os.path.abspath(os.path.join(sys.argv[0], *[os.pardir] * 2))
weight_dir = os.path.join(base_dir, "src", "weights", "cnn_classifier_weights.pth")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Asynchronous context manager to handle the startup and shutdown of the FastAPI application.

    Loads the model weights during startup and sets the model in the app state.
    Also handles cleanup during shutdown.
    """
    print(colored("STARTUP:".ljust(10) + "Loading model weights", "light_yellow"))
    app.state.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    app.state.model = ClassifierFactory.create(TClassifier.CNN, device=app.state.device)
    try:
        app.state.model.read_pth(weight_dir)
        print(colored("STARTUP:".ljust(10) + "Loading completed", "green"))
    except Exception as e:
        print(
            colored(
                "STARTUP:".ljust(10) + rf"Error while loading file {weight_dir}", "red"
            )
        )
        print(colored(f"Error: {str(e)}", "red"))
        print(colored("Traceback:", "red"))
        print(colored(traceback.format_exc(), "red"))
        app.state.model = None
    yield
    print(colored("SHUTDOWN:".ljust(10) + "Shutting down connection", "light_yellow"))


app = FastAPI(lifespan=lifespan)


@app.post(
    "/api/v1/predict",
    summary="Character classification",
    description="Character classification from gesture photo (the photo will be converted to grayscale and resized to 28x28 pixels)",
)
async def predict(image: UploadFile = File(...)) -> PredictionResponseDTO:
    """
    Predict the character in a given image.

    This endpoint receives an image file, processes it to grayscale, resizes it to 28x28 pixels,
    and returns the predicted class label and corresponding probabilities.

    :param image: The image file to be processed.
    :return: A PredictionResponseDTO object containing the prediction and probabilities.
    :raises HTTPException: If the model is unavailable, returns 503 Service Unavailable status.
    """
    if app.state.model is None:
        raise HTTPException(status_code=503, detail="Model is currently unavailable")

    img = await image.read()
    img_buffer = BytesIO(img)
    img_tensor = Converter.ImgToTensor(img_buffer)
    pred, probs_dict = app.state.model.predict(img_tensor)
    return {"prediction": pred, "probabilities": probs_dict}


if __name__ == "__main__":
    """Runs the FastAPI application with Uvicorn."""
    uvicorn.run("main:app", reload=True)
