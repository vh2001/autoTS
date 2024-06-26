import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch import nn
import numpy as np
from pathlib import Path
import torchvision.models as models
# create a tensor dataset


import src.config as cfg
from .BaseModelWrapper import BaseModelWrapper



def preprocess_vgg(images):
    """
    Preprocess images for VGG16 model.

    Args:
        images (np.ndarray): NumPy array of images.

    Returns:
        torch.Tensor: Preprocessed images as PyTorch tensor.   
    """
    images = torch.from_numpy(images)  # Convert NumPy array to PyTorch tensor
    images = images.unsqueeze(1)       # Add channel dimension
    images = images.repeat(1, 3, 1, 1) # Repeat the channel to mimic 3-channel RGB images
    return images




class VGG16_wrapper(BaseModelWrapper):
    def __init__(self, device=None):

        # initialize the VGG16 model
        self.model = models.vgg16()
        # check if cuda is available and use it if available
        self.device = device if device else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)

        

    def train_test_data(self, data):
        """
        Split data into train and test sets and return loaders.

        Args:
            data (List[Tuple[Any, Any]]): Data to split, each tuple is (X, y)..

        """
        print("Splitting data into train and test sets...")
        ts_data = []
        labels = []
        # split the data into features and labels
        for i in range(len(data)):
            ts_data.append(data[i][0])
            labels.append(data[i][1])

        # make images 3-channel as required by VGG16
        ts_data = preprocess_vgg(np.array(ts_data))
        data = list(zip(ts_data, labels))
      
        # check if we are using k-fold cross-validation
        if cfg.FOLDS == 0:
            dataset_size = len(data)
            test_size = int(dataset_size * cfg.DATA_SPLIT)
            train_size = dataset_size - test_size
            # split the data into train and test sets
            train_dataset, test_dataset = random_split(data, [train_size, test_size])
            # create data loaders based on config file
            train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=cfg.SHUFFLE)
            test_loader = DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False)

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
        torch.save(self.model.state_dict(), f"{path}/model.pt")

    def load_model(self, path):
        raise NotImplementedError("Loading model is not implemented for VGG16 model")
        # self.model.load_state_dict(torch.load(path, map_location=self.device))
