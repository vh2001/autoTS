import pandas as pd


def parse_lqe(df: pd.DataFrame)-> list[tuple[int, int, list[float]]]:
    """
    Parse LQE dataset into a list of tuples with the following format:
    (index, anomaly, timeseries)

    Anomalies:
    0: no anomaly
    1: suddenD
    2: suddenR
    3: instaD

    Parameters
    ----------
    df : LQE csv read into a pandas dataframe

    Returns
    -------
    list of tuples
    """
    
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
    
        anomalies.append((i, anomaly, ts))

    return anomalies
