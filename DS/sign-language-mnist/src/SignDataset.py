import pandas as pd
import torch
import torch.utils.data as data

class SignDataset(data.Dataset):
    def __init__(self, path):
        self.path = path
        self.data = pd.read_csv(path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        label = torch.tensor(row.iloc[0], dtype=torch.uint8)
        tensor = torch.tensor(row.iloc[1:].values, dtype=torch.float32).view(1, 28, 28) / 255.0        
        return tensor, label