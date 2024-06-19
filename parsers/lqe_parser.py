import pandas as pd
from pathlib import Path

def parse_lqe(path : Path)-> list[tuple[list[float], int]]:
    """
    Parse LQE dataset into a list of tuples with the following format:
    (timeseries, anomaly)

    Anomalies:
    0: no anomaly
    1: suddenD
    2: suddenR
    3: instaD

    Parameters
    ----------
    path : LQE csv path to read into a pandas dataframe

    Returns
    -------
    list of tuples
    """
    df = pd.read_csv(path)
    
    row = df.iloc[0]
    tab =  row.tolist()[1:]
    # array to store anomalies with labels
    anomalies = []
    # skip index
    for i in range(len(df)):
        row = df.iloc[i]
        tab = row.tolist()[1:]

        # timeseries data
        ts = tab[1:300]

        # anomaly label
        anomaly = int(tab[301])
    
        anomalies.append((ts, anomaly))

    return anomalies
