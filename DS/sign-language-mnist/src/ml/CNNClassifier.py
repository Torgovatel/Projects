import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class CNNClassifier(nn.Module):
    """
    A simple convolutional neural network (CNN) for image classification.

    The model consists of:
    - Two convolutional layers with ReLU activation and max pooling.
    - Two fully connected (linear) layers.

    :param output_size: Number of output classes.
    :type output_size: int

    :returns: A CNN model for image classification.
    :rtype: CNNClassifier
    """

    def __init__(self, output_size: int):
        """
        Initialize the CNN model with convolutional and linear layers.

        :param output_size: Number of output classes.
        :type output_size: int
        """
        super().__init__()
        self.conv1: nn.Conv2d = nn.Conv2d(
            in_channels=1, out_channels=32, kernel_size=3, padding=1
        )
        self.conv2: nn.Conv2d = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1
        )
        self.pool: nn.MaxPool2d = nn.MaxPool2d(kernel_size=2, stride=2)

        self.linear1: nn.Linear = nn.Linear(64 * 7 * 7, 128)
        self.linear2: nn.Linear = nn.Linear(128, output_size)

    def forward(self, x: Tensor) -> Tensor:
        """
        Perform a forward pass through the network.

        :param x: Input tensor of shape (batch_size, 1, 28, 28).
        :type x: Tensor

        :return: Output tensor of shape (batch_size, output_size), representing raw model outputs (scores).
        :rtype: Tensor
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
