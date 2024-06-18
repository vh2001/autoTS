import pandas as pd
import config as cfg
from transformation import transform
import torch
from torch.utils.data import DataLoader, TensorDataset
from autoTS.models.VGG16 import VGG16_wrapper

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


    # create dataset (X, y)
    data_labeled = [(transformed_data[i], data[i][1]) for i in range(len(transformed_data))]


    import torch.nn as nn
    # create a tensor dataset
    model = nn.model.VGG16()


    # create a model wrapper
    model_wrapper = VGG16_wrapper(model)

    # data split
    train_loader, test_loader = model_wrapper.train_test_data(data_labeled, cfg.BATCH_SIZE, cfg.SHUFFLE, cfg.TEST_SPLIT, cfg.FOLD)

    # train model
    model_wrapper.train(train_loader, cfg.EPOCHS, cfg.LR, None)

    # evaluate model
    predictions = model_wrapper.predict(test_loader)


    # get classification report
    from sklearn.metrics import classification_report
    df =  pd.Dataframe(classification_report(predictions, test_loader.y, output_dict=True)).T

    # save classification report
    df.to_csv(cfg.OUTPUT_PATH)

    


    









