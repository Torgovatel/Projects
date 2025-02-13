import string
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
    print(
        colored("STARTUP:".ljust(10), "light_yellow")
        + colored("Загрузка весов модели", "light_yellow")
    )
    app.state.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    app.state.model = ClassifierFactory.create(TClassifier.CNN, device=app.state.device)
    try:
        app.state.model.read_pth(weight_dir)
        print(colored("STARTUP:".ljust(10) + "Загрузка завершена", "green"))
    except Exception:
        print(
            colored("STARTUP:\t".ljust(10), "light_yellow")
            + colored(rf"Ошибка во время загрузки файла {weight_dir}", "red")
        )
        app.state.model = None
    yield
    print(
        colored("SHUTDOWN:".ljust(10), "light_yellow")
        + colored("Остановка соединения", "light_yellow")
    )


app = FastAPI(lifespan=lifespan)


@app.post(
    "/api/v1/predict",
    summary="Character classification",
    description="Character classification from gesture photo (the photo will be converted to grayscale and resized to 28x28 pixels)",
)
async def predict(image: UploadFile = File(...)) -> PredictionResponseDTO:
    if app.state.model is None:
        raise HTTPException(status_code=503, detail="Model is currently unavailable")

    img = await image.read()
    img_buffer = BytesIO(img)
    img_tensor = Converter.ImgToTensor(img_buffer)
    pred, probs_dict = app.state.model.predict(img_tensor)
    return {"prediction": pred, "probabilities": probs_dict}


if __name__ == "__main__":
    uvicorn.run("main:app", reload=True)
