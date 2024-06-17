from torch.utils.data import Dataset, DataLoader
import torch

class TimeSeriesDataset(Dataset):
    def __init__(self, data):
        """
        Args:
            data (list of tuples): A list where each tuple is (X, y) with
                                   X being the time series data and y being the label.
        """
        self.data = data
        # Convert list of tuples to two separate lists of tensors
        self.X = [torch.tensor(X, dtype=torch.float32) for X, _ in data]
        self.y = [torch.tensor(y, dtype=torch.long) for _, y in data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


