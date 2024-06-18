import pandas as pd
import src.config as cfg
from src.transformation import transform
import torch
from torch.utils.data import DataLoader, TensorDataset

import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    print(project_root)
    sys.path.insert(0, project_root)

from models.VGG16 import VGG16_wrapper

def preprocess_vgg(images):
    images = torch.from_numpy(images)  # Convert NumPy array to PyTorch tensor
    images = images.unsqueeze(1)       # Add channel dimension
    images = images.repeat(1, 3, 1, 1) # Repeat the channel to mimic 3-channel RGB images
    return images


def save_results(save_path, df):
    raise  NotImplementedError("Function not implemented yet")
    pass

if cfg.DATASET == "LQE":
    print(f"Running requested dataset: {cfg.DATASET}")
    # import parser
    from parsers.lqe_parser import parse_lqe
    df = pd.read_csv(cfg.DATASET_PATH)

    # parse data
    data = parse_lqe(df)

    # extract timeseries data and labels X,y
    ts_data = [x[2] for x in data]

    # import transformation
    print(f"Running requested transformation: {cfg.TRANSFORMATION}")
    transformed_data = transform(ts_data, cfg.TRANSFORMATION)

    # make images 3-channel
    transformed_data = preprocess_vgg(transformed_data)

    # create dataset (X, y)
    data_labeled = [(transformed_data[i], data[i][1]) for i in range(len(transformed_data))]


    import torchvision.models as models
    # create a tensor dataset
    model = models.vgg16(pretrained=False)

    # create a model wrapper
    model_wrapper = VGG16_wrapper(model)

    # data split
    train_loader, test_loader = model_wrapper.train_test_data(data_labeled, cfg.BATCH_SIZE, True, cfg.DATA_SPLIT, cfg.FOLDS)

    # train model
    model_wrapper.train(train_loader, cfg.EPOCHS, cfg.LR, [])

    # evaluate model
    predictions, ground_truth = model_wrapper.predict(test_loader)

    # get classification report
    from sklearn.metrics import classification_report
    df = pd.DataFrame(classification_report(predictions, ground_truth, output_dict=True)).T

    # create save folder if it doesnt exist
    save_path = f"results/{cfg.DATASET}/{cfg.MODEL}"
    if not os.path.exists(save_path):
        os.makedirs(save_path)



    # save classification report
    df.to_csv(f"{save_path}/classification_report.csv")














