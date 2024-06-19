import os
from pathlib import Path
import pandas as pd
import numpy as np
import json
import sys


import src.config as cfg

def parse_NILM(path: Path):
    """
    Parse NILM dataset into a list of tuples with the following format:
    (timeseries, appliance)

    Appliance is either appliance name or empty if the ts is empty
    Parameters
    ----------
    path : Path to NILM dataset

    Returns
    -------
    list of tuples
    """
    data = pd.read_pickle(path)

    # get all appliances
    appliances = set()
    for house, house_dict in data.items():
        for appliance in house_dict:
            appliances.add(appliance)
    

    # create a dictionary for the appliances to map the strings to integers, because pytorch does not support string labels
    appliances_dict = {}
    for i, a in enumerate(appliances):
        appliances_dict[a] = i

    # save appliances dict as json to be able to map the predictions back to the appliance names
    with open(f"{cfg.SAVE_PATH}/appliances_dict.json", "w") as f:
        json.dump(appliances_dict, f)

    
    timeseries = []
    # iterate over the houses and appliances in the dataset and create timeseries of length 300
    for house, house_dict in data.items():
        for appliance in house_dict:
            if "aggregate" in appliance:
                continue

            # load appliance dataframe
            df = house_dict[appliance]
            
            # get the number of timeseries
            n_timeseries = len(df) // 300
            for i in range(n_timeseries):
                cut_out = df[i*300:(i+1)*300].values.flatten()
                y = appliance
                # check if the timeseries is of the correct length
                if len(cut_out) != 300:
                    continue

                # check if the timeseries is not all zeros
                if np.all(cut_out == 0):
                    continue
                else:
                    timeseries.append((cut_out, appliances_dict[y]))

                # check if appliance is on 
    return timeseries