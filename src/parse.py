from pathlib import Path


from parsers.lqe_parser import parse_lqe
from parsers.ucr_parser import parse_ucr
from parsers.NILM_parser import parse_NILM



from parsers import * 
parsers = {
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
    data  = parsers[dataset](path)
    return data
    

