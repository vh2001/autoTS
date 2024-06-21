import pandas as pd
import src.config as cfg
from src.transformation import transform

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

    data = parse_data(cfg.DATASET, cfg.DATASET_PATH)

    if cfg.TRANSFORMATION is not None:
        ts_data = [x[0] for x in data]

        transformed_data = transform(ts_data, cfg.TRANSFORMATION)


        data = [(transformed_data[i], data[i][1]) for i in range(len(transformed_data))]

    run_model(data)


main()










