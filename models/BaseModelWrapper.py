from typing import Any, List, Tuple, Callable, Iterable
from abc import ABC, abstractmethod

class BaseModelWrapper(ABC):
    @abstractmethod
    def train_test_data(self, data: List[Tuple[Any, Any]]) -> Tuple[Iterable, Iterable]:
        """
        Abstract method to split data into train and test sets and return loaders or iterators depending on the framework.
        Arguments such as batch size, shuffle, etc. should be defined in the config file and used here.
        
        Args:
            data (List[Tuple[Any, Any]]): Data to split, each tuple is (X, y).
            
        Returns:
            Tuple[Iterable, Iterable]: Training and testing data loaders or iterators.
        """
        pass

    @abstractmethod
    def train(self, train_loader: Iterable, epochs: int, callbacks: List[Callable[[Any, int, int], None]] = [], *args, **kwargs) -> None:
        """
        Abstract method to train the model on given data. The trained model should be saved in the self.model attribute.
        
        Args:
            train_loader (Iterable): Data loader or iterable to train on.
            epochs (int): Number of training epochs.
            callbacks (List[Callable[[Any, int, int], None]]): Optional callbacks to execute during training.
        """
        pass

    @abstractmethod
    def predict(self, test_loader: Iterable, *args, **kwargs) -> Any:
        """
        Abstract method to predict on given data loader and return predictions. 
        The function should return the predictions and true labels as a tuple (Y_pred, Y_true)
        
        Args:
            test_loader (Iterable): Data loader or iterable to predict on.
        
        Returns:
            Y_pred: Predictions on the test data.
            Y_true: True labels of the test data.
        """
        pass

    @abstractmethod
    def save_model(self, path: str, *args, **kwargs) -> None:
        """
        Abstract method to save the model to the specified path. This
        
        Args:
            path (str): File path to save the model.
        """
        pass

    def load_model(self, path: str, *args, **kwargs) -> None:
        """
        Abstract method to load a model from the specified path.
        This is an optional method and can be implemented if needed.
        Args:
            path (str): File path from which to load the model.
        """
        pass
