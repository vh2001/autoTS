import pandas as pd
import config as cfg
from transformation import transform
import torch
from torch.utils.data import DataLoader, TensorDataset
from models.models import load_model, evaluate


if cfg.DATASET == "LQE":
    print(f"Running requested dataset: {cfg.DATASET}")
    # import parser
    from parsers.lqe_parser import parse_lqe
    df = pd.read_csv(cfg.DATASET_DIR + "/LQE.csv")

    # parse data
    data = parse_lqe(df)

    # extract timeseries data and labels X,y
    ts_data = [(x[2], x[1]) for x in data]

    # import transformation
    print(f"Running requested transformation: {cfg.TRANSFORMATION}")
    transformed_data = transform(ts_data, cfg.TRANSFORMATION)



    # create a dataset
    dataset = TensorDataset(torch.Tensor(transformed_data))

    # load model
    model = load_model(cfg.MODEL)

    # evaluate model
    evaluate(cfg.EPOCHS, cfg.LR, cfg.BATCH_SIZE, model, dataset, cfg.FOLDS, cfg.DATA_SPLIT)
   

    ###


    




    









