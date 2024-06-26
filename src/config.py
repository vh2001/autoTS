import json
from datetime import datetime
import os
DATASET = "NILM" # select DATASET for any NILM data set put NILM and then provide the direct path to the .pkl file
DATASET_PATH = "../Energy_graph/energy-knowledge-graph/data/parsed/DRED.pkl"

TRANSFORMATION = "GADF" # select transformation and keep in mind which transformation works with which model the available transformations can be seen in the transformation.py file
MODEL = "VGG16" # select model the available models can be found on our github 
TASK ="classification" # for now only classification is supported


EXPERIMENT_NAME = f"{DATASET}_{TRANSFORMATION}_{MODEL}" # name of expirement

# add date to the experiment name
EXPERIMENT_NAME = f"{EXPERIMENT_NAME}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}" # we add the date to the experiment name by default this can be changed here

SAVE_PATH = f"./results/{EXPERIMENT_NAME}" # save path where the results and model will be saved


# create save folder if it doesnt exist
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)


# Training parameters
DATA_SPLIT=0.8 # 80% train, 20% test , If FOLDS is larger than 1 this is ignored !!!!!!!!!!!
FOLDS = 0 # if not using cross validation, set to 0 and DATA_SPLIT will be used
SHUFFLE = True # shuffle training data


# model parameters
EPOCHS = 2
LR = 0.01
BATCH_SIZE = 32
CALLBACKS = []


# InceptionTime specific parameters (for more details read paper: https://arxiv.org/abs/1909.04939)
INPUT_SHAPE = (299,1) # input shape for the InceptionTime model for eg. (300, 1) for 300 timesteps and 1 feature (univariate time series
NUM_CLASSES = 5 # number of classes for classification
DEPTH = None # depth of the InceptionTime model leave None for default(6)
KERNEL_SIZE = None # kernel size for the InceptionTime model leave None for default(42)


# function to save the used config
def save_config():
    
    if MODEL == "InceptionTime":
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
        "CALLBACKS": CALLBACKS,

        "INPUT_SHAPE": INPUT_SHAPE,
        "NUM_CLASSES": NUM_CLASSES,
        "DEPTH": DEPTH,
        "KERNEL_SIZE": KERNEL_SIZE
        }
    else:
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
            
            
        
    }

    with open(f"{SAVE_PATH}/config.json", "w") as f:
        json.dump(cfg, f)