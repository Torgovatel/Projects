import torch.nn as nn
import torch.nn.functional as F

class CNNClassifier(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # После двух пулингов 28×28 → 14×14 → 7×7
        self.fc2 = nn.Linear(128, output_size)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 28×28 → 14×14
        x = self.pool(F.relu(self.conv2(x)))  # 14×14 → 7×7
        x = x.view(x.size(0), -1)  # Преобразование в плоский вектор
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
