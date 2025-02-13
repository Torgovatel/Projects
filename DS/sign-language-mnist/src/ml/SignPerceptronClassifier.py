import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SignPerceptronClassifier(nn.Module):
    """
    A simple two-layer perceptron for sign language classification.

    This model consists of:
    - A fully connected hidden layer with ReLU activation.
    - A fully connected output layer.

    The input is expected to be a grayscale image tensor of shape (batch_size, 1, 28, 28).
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """
        Initialize the perceptron model with linear layers.

        :param input_size: Number of input features (e.g., 28x28 = 784 for images).
        :param hidden_size: Number of neurons in the hidden layer.
        :param output_size: Number of output classes.
        """
        super().__init__()
        self.linear1: nn.Linear = nn.Linear(input_size, hidden_size)
        self.linear2: nn.Linear = nn.Linear(hidden_size, output_size)

    def forward(self, x: Tensor) -> Tensor:
        """Perform a forward pass through the network.

        :param x: Input tensor of shape (batch_size, input_size).
        :return: Output tensor of shape (batch_size, output_size), representing raw model outputs (scores).
        """
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x
