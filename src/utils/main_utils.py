import os
import sys
import yaml
import numpy as np
import dill
from src.logger import logger
from src.exception import CustomException

def read_yaml_file(file_path: str) -> dict:
    """
    Reads a YAML file and returns its content as a dictionary.
    """
    try:
        logger.info(f"Reading YAML file from path: {file_path}")
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise CustomException(e, sys) from e

def write_yaml_file(file_path: str, content: dict, replace: bool = False) -> None:
    """
    Writes a dictionary to a YAML file. If replace is True, deletes the old file if it exists.
    """
    try:
        logger.info(f"Writing YAML file to path: {file_path}")
        if replace and os.path.exists(file_path):
            os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as yaml_file:
            yaml.dump(content, yaml_file)
    except Exception as e:
        raise CustomException(e, sys) from e

def save_object(file_path: str, obj: object) -> None:
    """
    Saves a python object using dill serialization.
    """
    try:
        logger.info(f"Saving serialized object to path: {file_path}")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys) from e

def load_object(file_path: str) -> object:
    """
    Loads a serialized python object using dill deserialization.
    """
    try:
        logger.info(f"Loading serialized object from path: {file_path}")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys) from e

def save_numpy_array_data(file_path: str, array: np.ndarray) -> None:
    """
    Saves a numpy array to a binary file.
    """
    try:
        logger.info(f"Saving numpy array to path: {file_path}")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise CustomException(e, sys) from e

def load_numpy_array_data(file_path: str) -> np.ndarray:
    """
    Loads a numpy array from a binary file.
    """
    try:
        logger.info(f"Loading numpy array from path: {file_path}")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys) from e
