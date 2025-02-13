import string
import torch
import numpy as np
from ml.CNNClassifier import CNNClassifier
from ml.SignPerceptronClassifier import SignPerceptronClassifier
from typing import TypeAlias, Tuple, Dict, Literal
from PIL import Image
from io import BytesIO
from enum import Enum

ClassifierModel: TypeAlias = CNNClassifier | SignPerceptronClassifier

ClassLabels = list(string.ascii_uppercase[:26])

ProbabilityDict = Dict[str, float]


class TClassifier(Enum):
    CNN = "cnn"
    PERCEPTRON = "perceptron"


class ClassifierAdapter:
    def __init__(self, model: ClassifierModel, device: torch.device = None):
        self.model = model
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)
        self.model.eval()

    def read_pth(self, path: str):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def predict(self, X: torch.Tensor) -> Tuple[str, ProbabilityDict]:
        with torch.no_grad():
            X = X.to(self.device)
            pred = self.model(X)
        probs = torch.softmax(pred, dim=1).cpu().numpy().squeeze().tolist()
        return ClassLabels[pred.argmax().item()], dict(zip(ClassLabels, probs))


class Converter:
    @classmethod
    def ImgToTensor(cls, image: BytesIO) -> torch.Tensor:
        try:
            img = Image.open(image).convert("L").resize((28, 28))
        except Exception:
            raise ValueError("Invalid image file.")
        img_arr = np.array(img) / 255.0
        return torch.tensor(img_arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    @classmethod
    def TensorToImg(cls, tensor: torch.Tensor) -> Image.Image:
        if tensor.dim() == 3 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
        elif tensor.ndim != 2:
            raise ValueError(
                f"Invalid tensor shape: {tuple(tensor.shape)}, need (1, 28, 28) or (28, 28)"
            )
        img_arr = (tensor.numpy() * 255).clip(0, 255).astype(np.uint8)
        return Image.fromarray(img_arr, mode="L")


class ClassifierFactory:
    @classmethod
    def create(
        cls, model_name: TClassifier, device: torch.device = None
    ) -> ClassifierAdapter:
        if model_name == TClassifier.CNN:
            model = CNNClassifier(output_size=25)
        elif model_name == TClassifier.PERCEPTRON:
            model = SignPerceptronClassifier(
                input_size=28**2, hidden_size=128, output_size=25
            )
        return ClassifierAdapter(model, device=device)
