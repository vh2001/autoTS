from pathlib import Path

import importlib


def parse_data(dataset:str, path:Path):
    """
    Parse data using the specified parser

    Parameters
    ----------
    dataset : str
        Dataset to parse
    path : str
        Path to dataset
    """
    # import parser module
    parser_module = importlib.import_module(f"parsers.{dataset}_parser")

    # get parser function
    parser_function = getattr(parser_module, "parse")

    data = parser_function(path)
    return data
    

