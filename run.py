import pandas as pd
import src.config as cfg
from src.transformation import transform
import torch
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import sys
import os
import numpy as np
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    print(project_root)
    sys.path.insert(0, project_root)

from autoTS.models.wrapper.VGG16_wrapper import VGG16_wrapper

from src.parse import parse_data
from src.run_model import run_model






def main():
    cfg.save_config()

    # if cfg.DATASET == "UCR":
    #     data_dict = parse_data(cfg.DATASET, cfg.DATASET_PATH)
    #     reports = []
    #     # parse data
    #     f1s = []
    #     accs = []


    #     for k, data in data_dict.items():
    #         train_data = data["train"]
    #         test_data = data["test"]
    #         # extract timeseries data and labels X,y
    #         ts_data_train = [x[0] for x in train_data]
    #         ts_data_test = [x[0] for x in test_data]

    #         print(f"SHAPES train: {ts_data_train.shape} test: {ts_data_test.shape}")
    #         # import transformation
    #         print(f"Running requested transformation: {cfg.TRANSFORMATION}")
    #         transformed_data_train = transform(ts_data_train, cfg.TRANSFORMATION)
    #         transformed_data_test = transform(ts_data_test, cfg.TRANSFORMATION)

    #         # make images 3-channel
    #         transformed_data_train = preprocess_vgg(transformed_data_train)
    #         transformed_data_test = preprocess_vgg(transformed_data_test)

    #         # create dataset (X, y)
    #         data_labeled_train = [(transformed_data_train[i], train_data[i][1]) for i in range(len(transformed_data_train))]
    #         data_labeled_test = [(transformed_data_test[i], test_data[i][1]) for i in range(len(transformed_data_test))]



    #         # create a tensor dataset
    #         model = models.vgg16(pretrained=False)

    #         # create a model wrapper
    #         model_wrapper = VGG16_wrapper(model)


    #         # create train loader
    #         train_loader = DataLoader(TensorDataset(*zip(*data_labeled_train)), batch_size=cfg.BATCH_SIZE, shuffle=True)
    #         test_loader = DataLoader(TensorDataset(*zip(*data_labeled_test)), batch_size=cfg.BATCH_SIZE, shuffle=False)



    #         # train model
    #         model_wrapper.train(train_loader, cfg.EPOCHS, cfg.LR, [])

    #         # save model
    #         # model_wrapper.save_model(f"{sa}/{k}")

    #         # evaluate model
    #         predictions, ground_truth = model_wrapper.predict(test_loader)

    #         # get classification report
    #         from sklearn.metrics import classification_report
    #         cr = pd.DataFrame(classification_report(predictions, ground_truth, output_dict=True))

    #         f1 = cr["macro avg"][2]
    #         acc = cr["accuracy"][0]
    #         f1s.append(f1)
    #         accs.append(acc)
    #         reports.append((k, cr))

    #         # save classification report
    #         cr.to_csv(f"{save_path}/{k}_classification_report.csv")
    #         # create save folder if it doesnt exist

    #     # save f1s and accuracies as npy
    #     f1s = np.array(f1s)
    #     accs = np.array(accs)

    #     np.save(f"{save_path}/f1s.npy", f1s)
    #     np.save(f"{save_path}/accs.npy", accs)
        

    data = parse_data(cfg.DATASET, cfg.DATASET_PATH)

    if cfg.TRANSFORMATION is not None:
        ts_data = [x[0] for x in data]

        transformed_data = transform(ts_data, cfg.TRANSFORMATION)


        data = [(transformed_data[i], data[i][1]) for i in range(len(transformed_data))]

    run_model(data)


main()










