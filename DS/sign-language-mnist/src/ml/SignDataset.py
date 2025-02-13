import pandas as pd
import torch
import torch.utils.data as data
from typing import Tuple


class SignDataset(data.Dataset):
    """
    A dataset for sign language classification, loading images from a CSV file.

    Each sample consists of a 28x28 grayscale image and a corresponding label.
    The first column in the CSV file contains the label, while the remaining
    columns represent pixel values of the image in row-major order.

    The pixel values are normalized to the range [0, 1] and reshaped to (1, 28, 28).
    """

    def __init__(self, path: str):
        """Initialize the dataset by loading data from a CSV file.

        :param path: Path to the CSV file containing the dataset.
        """
        self.path = path
        self.data = pd.read_csv(path)

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[int, torch.Tensor]:
        """Retrieve the image and label at the given index.

        :param idx: Index of the sample.
        :return: A tuple (tensor, label), where:
        - tensor (torch.Tensor): A 28x28 image tensor of shape (1, 28, 28), normalized to [0, 1].
        - label (torch.Tensor): A tensor containing the class label as uint8.
        """
        row = self.data.iloc[idx]
        label = torch.tensor(row.iloc[0], dtype=torch.uint8)
        tensor = (
            torch.tensor(row.iloc[1:].values, dtype=torch.float32).view(1, 28, 28)
            / 255.0
        )
        return tensor, label
