from pathlib import Path


from parsers import lqe_parser, ucr_parser, NILM_parser




parsers = {
    "lqe" : lqe_parser.parse_lqe,
    "ucr" : ucr_parser.parse_ucr,
    "NILM" : NILM_parser.parse_NILM
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
    return parsers[dataset](path)
    

