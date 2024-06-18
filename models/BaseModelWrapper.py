from typing import Any, List, Tuple, Callable, Iterable
from abc import ABC, abstractmethod

class BaseModelWrapper(ABC):
    @abstractmethod
    def train_test_data(self, data: List[Tuple[Any, Any]], batch_size: int, shuffle: bool = True, test_split: float = 0.2, fold: int = 0) -> Tuple[Iterable, Iterable]:
        """
        Abstract method to split data into train and test sets and return loaders or iterators depending on the framework.
        
        Args:
            data (List[Tuple[Any, Any]]): Data to split, each tuple is (X, y).
            batch_size (int): Batch size for loading data.
            shuffle (bool): Whether to shuffle the data before splitting.
            test_split (float): Fraction of the data to be used as the test set (used if fold=0).
            fold (int): Specifies which fold to use as the test set if using k-fold cross-validation.

        Returns:
            Tuple[Iterable, Iterable]: Training and testing data loaders or iterators.
        """
        pass

    @abstractmethod
    def train(self, train_loader: Iterable, epochs: int, callbacks: List[Callable[[Any, int, int], None]] = [], *args, **kwargs) -> None:
        """
        Abstract method to train the model on given data loader.
        
        Args:
            train_loader (Iterable): Data loader or iterable to train on.
            epochs (int): Number of training epochs.
            callbacks (List[Callable[[Any, int, int], None]]): Optional callbacks to execute during training.
        """
        pass

    @abstractmethod
    def evaluate(self, test_loader: Iterable, *args, **kwargs) -> Any:
        """
        Abstract method to evaluate the model using a data loader and return evaluation metrics.
        
        Args:
            test_loader (Iterable): Data loader or iterable for evaluation.
        
        Returns:
            Any: Evaluation metrics.
        """
        pass

    @abstractmethod
    def predict(self, test_loader: Iterable, *args, **kwargs) -> Any:
        """
        Abstract method to predict on given data loader and return predictions.
        
        Args:
            test_loader (Iterable): Data loader or iterable to predict on.
        
        Returns:
            Any: Predictions made by the model.
        """
        pass

    @abstractmethod
    def save_model(self, path: str, *args, **kwargs) -> None:
        """
        Abstract method to save the model to the specified path.
        
        Args:
            path (str): File path to save the model.
        """
        pass

    @abstractmethod
    def load_model(self, path: str, *args, **kwargs) -> None:
        """
        Abstract method to load a model from the specified path.
        
        Args:
            path (str): File path from which to load the model.
        """
        pass
