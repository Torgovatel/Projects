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
    """
    Enum for classifier types.

    :param CNN: Refers to the CNN classifier model.
    :param PERCEPTRON: Refers to the perceptron classifier model.

    :result: Returns an enum value representing the classifier type.
    """

    CNN = "cnn"
    PERCEPTRON = "perceptron"


class ClassifierAdapter:
    """Adapter class for loading and using classifiers."""

    def __init__(self, model: ClassifierModel, device: torch.device = None):
        """
        Initialize the adapter with classifier model and torch device.

        :param model: The model to be used for predictions (CNN or Perceptron).
        :param device: The device on which the model should run (default is CUDA or CPU).
        """
        self.model = model
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)
        self.model.eval()

    def read_pth(self, path: str):
        """
        Load model weights from a given file path.

        :param path: Path to the model weights file.
        """
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def predict(self, X: torch.Tensor) -> Tuple[str, ProbabilityDict]:
        """
        Make a prediction using the classifier model.

        :param X: Input tensor to make the prediction on shape: (batch_size, channels, height, width).

        :return: Returns a tuple consisting of:
        - label (str): The predicted class.
        - ProbabilityDict (Dict[str, float]): A dictionary of class probabilities.
        """
        with torch.no_grad():
            X = X.to(self.device)
            pred = self.model(X)
        probs = torch.softmax(pred, dim=1).cpu().numpy().squeeze().tolist()
        return ClassLabels[pred.argmax().item()], dict(zip(ClassLabels, probs))


class Converter:
    """A utility class for converting between image formats and tensors."""

    @classmethod
    def ImgToTensor(cls, image: BytesIO) -> torch.Tensor:
        """
        Convert an image (in BytesIO format) to a tensor.

        :param image: The image in BytesIO format.

        :return: Returns a tensor representing the grayscale image of shape (1, 1, 28, 28),
                 with pixel values normalized to the range [0, 1] and of type torch.float32.
        """
        try:
            img = Image.open(image).convert("L").resize((28, 28))
        except Exception:
            raise ValueError("Invalid image file.")
        img_arr = np.array(img) / 255.0
        return torch.tensor(img_arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    @classmethod
    def TensorToImg(cls, tensor: torch.Tensor) -> Image.Image:
        """
        Convert a tensor to an image.

        :param tensor: The tensor to be converted into an image (shape: (1, 28, 28) or (28, 28)).

        :return: Returns a PIL image of the tensor.
        """
        if tensor.dim() == 3 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
        elif tensor.ndim != 2:
            raise ValueError(
                f"Invalid tensor shape: {tuple(tensor.shape)}, need (1, 28, 28) or (28, 28)"
            )
        img_arr = (tensor.numpy() * 255).clip(0, 255).astype(np.uint8)
        return Image.fromarray(img_arr, mode="L")


class ClassifierFactory:
    """Factory class for creating classifier adapters."""

    @classmethod
    def create(
        cls, model_name: TClassifier, device: torch.device = None
    ) -> ClassifierAdapter:
        """
        Creates an adapter for the specified classifier model.

        :param model_name: The classifier model to use (either CNN or Perceptron).
        :param device: The device on which the model should run (default is CUDA or CPU).

        :return: Returns a ClassifierAdapter instance initialized with the specified model.
        """
        if model_name == TClassifier.CNN:
            model = CNNClassifier(output_size=25)
        elif model_name == TClassifier.PERCEPTRON:
            model = SignPerceptronClassifier(
                input_size=28**2, hidden_size=128, output_size=25
            )
        return ClassifierAdapter(model, device=device)
