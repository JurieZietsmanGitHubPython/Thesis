import logging
import time
from functools import wraps
from typing import Dict, Union

import numpy


def create_logger(name, stream: bool = False, file: bool = False, filename: str = "HGSADC Logs.log",
                  log_format: str = "%(asctime)s:%(levelname)s:%(name)s:%(funcName)s:%(message)s",
                  level: Union[int, str] = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)

    if stream:
        handler = logging.StreamHandler()
        _configure_and_add_handler(logger, handler, log_format, level)

    if file:
        handler = logging.FileHandler(filename)
        _configure_and_add_handler(logger, handler, log_format, level)

    return logger


def _configure_and_add_handler(logger: logging.Logger, handler: Union[logging.StreamHandler, logging.FileHandler],
                               log_format: str, level: Union[int, str]):
    handler.setFormatter(logging.Formatter(log_format))
    handler.setLevel(level)
    logger.addHandler(handler)


def time_this(func):
    """Small helper function to time some methods.

    Copied from jjupe's git."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        r = func(*args, **kwargs)
        end = time.perf_counter()
        logging.info(f'Runtime: {end - start}')
        return r

    return wrapper


def rank_list(to_rank: list, ascending: bool = True) -> list:
    """Uses NumPy to rank a numerical list.

    Algorithm used here is from
    https://stackoverflow.com/questions/5284646/rank-items-in-an-array-using-python-numpy-without-sorting-array-twice"""
    array = numpy.array(to_rank)

    # Flip all values' signs to sort descending
    if not ascending:
        array = -array

    # argsort returns the an array showing the indices of elements in their ranked order
    # e.g. [5, 8, 2] will return [2, 0, 1]
    temp = array.argsort()

    # Convert the sorted indices into an array of the ranks of each element
    ranks = numpy.empty_like(temp)
    # e.g. [ , , ][2, 0, 1] = [0, 1, 2] gives [1, 2, 0]
    ranks[temp] = numpy.arange(len(array))

    return list(ranks)


def append_to_chromosome(chromosome: Dict[int, list], key: int, value, add_duplicates: bool = True):
    """Appends value to list at key.

    First checks if chromosome has a list at the key. If not, then creates a list."""
    # if should_print and len(chromosome.keys()) == 0 and (key is None or value is None):
    #     print(f"Blank chromosome and value/key. Adding {value} to {key}.")

    if chromosome.get(key) is None:
        chromosome[key] = []
    if add_duplicates or value not in chromosome[key]:
        chromosome[key].append(value)
