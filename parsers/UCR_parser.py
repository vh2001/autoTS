import pandas as pd
import os 
from pathlib import Path
def parse(path : Path)-> dict:
    datasets = {}
    for dataset in os.listdir(path):
        if dataset == "Missing_value_and_variable_length_datasets_adjusted":
            print("Skipping", dataset)
            continue
        curr_dict = {}
        for f in os.listdir(path / dataset):
            
            if f.endswith(".tsv"):
                df = pd.read_csv(path / dataset/f, sep="\t", header=None)
                
                if "TEST"in f:
                    y = df[0].values
                    X = df.drop(0, axis=1).values
                    curr_dict["test"] = (X,y)

                elif "TRAIN" in f:
                    y = df[0].values
                    X = df.drop(0, axis=1).values
                    curr_dict["train"] = (X,y)
                else:
                    print("Skipping", f)
                    continue
        datasets[dataset] = curr_dict

    return datasets