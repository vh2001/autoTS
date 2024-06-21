from tensorflow import keras
import tensorflow as tf 
from sklearn.model_selection import train_test_split

from .BaseModelWrapper import BaseModelWrapper
from .InceptionTime import Classifier_INCEPTION
import src.config as cfg
import os
class InceptionTime_wrapper(BaseModelWrapper):

    def __init__(self, device=None):
        """
        Initialize the InceptionTime model.

        Args:
            device (str): Device to use for training and inference.
        """
        # check if cuda is available with tensorflow
        if not tf.test.is_gpu_available():
            raise Exception("No GPU found, please use a GPU to train the model.")
        else :
            print("GPU found, using GPU for training the model.")

        if cfg.INPUT_SHAPE is None:
            raise Exception("Input shape must be provided in the config.py file for the InceptionTime model.")

        if cfg.NUM_CLASSES is None:
            raise Exception("Number of classes must be provided in the config.py file for the InceptionTime model.")
        if not os.exists(f"{cfg.SAVE_PATH}/model/"):
            os.makedirs(f"{cfg.SAVE_PATH}/model/")


        # define callbacks here
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.25, patience=7, min_lr=0.0001)
        early_stopping = keras.callbacks.EarlyStopping(patience=9, restore_best_weights=True, monitor='loss')

        self.model = Classifier_INCEPTION(
            output_directory=f"{cfg.SAVE_PATH}/model/",
            input_shape=cfg.INPUT_SHAPE,
            nb_classes=cfg.NUM_CLASSES,
            verbose=True,
            build=True,
            batch_size=cfg.BATCH_SIZE,
            nb_epochs=cfg.EPOCHS,
            lr=cfg.LR,
            depth=6 if cfg.DEPTH is None else cfg.DEPTH,
            kernel_size=41 if cfg.KERNEL_SIZE is None else cfg.KERNEL_SIZE,
            callbacks=[reduce_lr, early_stopping],
            # change activation and loss function if you dont do multiclass classification
            loss_fn='softmax',
            activation='categorical_crossentropy'

            )


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
        X = []
        y = []
        for i in range(len(data)):
            X.append(data[i][0])
            y.append(data[i][1])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=cfg.DATA_SPLIT, shuffle=cfg.SHUFFLE,
                                                        random_state=42)  

        return zip(X_train, y_train), zip(X_test, y_test)

    def train(self, train_data, epochs, lr,  callbacks=[]):
        """
        Train the model on given data loader.
        """
        print("Starting training...")
        X, Y = zip(*train_data)

        self.model.model.fit(X, Y, epochs=epochs, batch_size=cfg.BATCH_SIZE, callbacks=callbacks)
        


    
    def predict(self, data):
        """
        Predict on the given data.
        """
        print("Predicting...")
        X,y = zip(*data)

        predictions = self.model.model.predict(X)

        return predictions, y
    
    

    def save_model(self, path):
        print("Saving model...")
        self.model.save(path)
    
    def load_model(self, path):
        print("Loading model...")
        model = keras.models.load_model(path)