

DATASET = "LQE"
DATASET_PATH = "datasets/LQE/dataset_LQE_uncut.csv"
TRANSFORMATION = "GADF"
MODEL = "vgg11"



# Training parameters
DATA_SPLIT=0.8 # 80% train, 20% test , If FOLDS is larger than 1 this is ignored !!!!!!!!!!!
FOLDS = 1 # if not using cross validation, set to 1 and DATA_SPLIT will be used

# model parameters
EPOCHS = 10
LR = 0.01
BATCH_SIZE = 32



