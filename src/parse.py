from pathlib import Path


# import parsers ADD NEW PARSERS HERE
from parsers.lqe_parser import parse_lqe
from parsers.ucr_parser import parse_ucr
from parsers.NILM_parser import parse_NILM


# add parser to the parsers dictionary ADD NEW PARSERS HERE
parsers_dict = {
    "LQE" : parse_lqe,
    "UCR" : parse_ucr,
    "NILM" : parse_NILM
}

def parse_data(dataset:str, path:Path):
    """
    Parse data using the specified parser

    Parameters
    ----------
    dataset : str
        Dataset to parse
    path : str
    """
    data  = parsers_dict[dataset](path)
    return data
    

