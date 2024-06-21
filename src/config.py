import json
from datetime import datetime
import os
DATASET = "LQE"
DATASET_PATH = "./data/LQE/dataset_LQE_uncut.csv"

TRANSFORMATION = "GADF"
MODEL = "vgg11"
TASK ="classification" # for now only classification is supported


EXPERIMENT_NAME = f"{DATASET}_{TRANSFORMATION}_{MODEL}"

# add date to the experiment name
EXPERIMENT_NAME = f"{EXPERIMENT_NAME}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
SAVE_PATH = f"./results/{EXPERIMENT_NAME}"


# create save folder if it doesnt exist
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)


# Training parameters
DATA_SPLIT=0.8 # 80% train, 20% test , If FOLDS is larger than 1 this is ignored !!!!!!!!!!!
FOLDS = 0 # if not using cross validation, set to 0 and DATA_SPLIT will be used
SHUFFLE = True # shuffle training data


# model parameters
EPOCHS = 10
LR = 0.01
BATCH_SIZE = 32
CALLBACKS = []


# InceptionTime specific parameters (for more details read paper: https://arxiv.org/abs/1909.04939)
INPUT_SHAPE = None # input shape for the InceptionTime model for eg. (300, 1) for 300 timesteps and 1 feature (univariate time series
NUM_CLASSES = None # number of classes for classification
DEPTH = None # depth of the InceptionTime model 
KERNEL_SIZE = None # kernel size for the InceptionTime model


def save_config():
    cfg = {
        "DATASET": DATASET,
        "DATASET_PATH": DATASET_PATH,
        "TRANSFORMATION": TRANSFORMATION,
        "MODEL": MODEL,
        "TASK": TASK,

        "DATA_SPLIT": DATA_SPLIT,
        "FOLDS": FOLDS,
        "SHUFFLE": SHUFFLE,

        "EPOCHS": EPOCHS,
        "LR": LR,
        "BATCH_SIZE": BATCH_SIZE,
        "CALLBACKS": CALLBACKS
    }

    with open(f"{SAVE_PATH}/config.json", "w") as f:
        json.dump(cfg, f)