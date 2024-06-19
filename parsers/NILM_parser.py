import os
from pathlib import Path
import pandas as pd
import numpy as np

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
    
    timeseries = []
    count_empty = 0
    for house, house_dict in data.items():
        for appliance in house_dict:
            if "aggregate" in appliance:
                continue

            # split the df into 300 length timeseries
            df = house_dict[appliance]
            
            # get the number of timeseries
            n_timeseries = len(df) // 300
            for i in range(n_timeseries):
                cut_out = df[i*300:(i+1)*300].values
                y = appliance
                # check if the timeseries is of the correct length
                if len(cut_out) != 300:
                    continue

                # check if the timeseries is not all zeros
                if np.all(cut_out == 0):
                    count_empty+=1
                    y = "empty"
                else:
                    timeseries.append((cut_out, y))

                # check if appliance is on 
    return timeseries