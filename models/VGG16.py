import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch import nn
from .BaseModelWrapper import BaseModelWrapper



class VGG16_wrapper(BaseModelWrapper):
    def __init__(self, model, device=None):
        self.model = model
        self.device = device if device else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)

        

    def train_test_data(self, data, batch_size, shuffle=True, test_split=0.2, fold=0):
        """
        Split data into train and test sets and return loaders.

        Args:
            data (List[Tuple[Any, Any]]): Data to split, each tuple is (X, y).
            batch_size (int): Batch size for loading data.
            shuffle (bool): Whether to shuffle the data before splitting.
            test_split (float): Fraction of the data to be used as the test set.
            fold (int): Specifies which fold to use as the test set if using k-fold cross-validation.

        """
        print("Splitting data into train and test sets...")



        if fold == 0:
            dataset_size = len(data)
            test_size = int(dataset_size * test_split)
            train_size = dataset_size - test_size
            train_dataset, test_dataset = random_split(data, [train_size, test_size])
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            return train_loader, test_loader
        else:
            raise NotImplementedError("K-fold cross-validation is not implemented")

    def train(self, train_loader, epochs, lr,  callbacks=[]):
        """
        Train the model on given data loader.

        Args:
            train_loader (Iterable): Data loader to train on.
            epochs (int): Number of training epochs.
            callbacks  Optional callbacks to execute during training.
        """
        print("Starting training...")
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        self.model.train()
        for epoch in range(epochs):
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device).float(), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            # Callbacks could be used here to log metrics or save intermediate models
            for callback in callbacks:
                callback(self.model, epoch)

    def predict(self, test_loader):
        """
        Predict on given data loader and return predictions.

        Args:
            test_loader (Iterable): Data loader to predict on.
        Returns:
            List: Predictions made by the model.
            List: Ground truth labels.
        """
        print("Making predictions...")
        self.model.eval()
        predictions = []
        ground_truth = []
        with torch.no_grad():
            for inputs, gt in test_loader:
                inputs = inputs.to(self.device).float()
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                predictions.extend(predicted.cpu().numpy())
                ground_truth.extend(gt.cpu().numpy())
        return predictions, ground_truth

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
