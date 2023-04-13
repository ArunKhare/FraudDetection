import os
import numpy as np
import pandas as pd
import dill, yaml, json, joblib
from box import ConfigBox
from pathlib import Path
from typing import Any
from ensure import ensure_annotations

from box.exceptions import BoxValueError
from fraudDetection.exception import fraudDetectionException
from fraudDetection.logger import logging
from fraudDetection.constants import DATASET_SCHEMA_COUMNS_KEY

@ensure_annotations 
def create_directories(path_to_directories:list,verbose=True):
    """ create list of directories
    Args: 
        path_to_directories(list): list of path of directories
        ignore_log (bool,optional): ignore if multiple dirs is to be created. Defaults to False.
    """
    try:
        for path in path_to_directories:
            os.makedirs(path,exist_ok=True)
            if verbose:
                logging.info("created directories at:{path}")
    except Exception as e:
        raise fraudDetectionException(e,sys) from e
    
@ ensure_annotations
def read_yaml(path_to_yaml:Path) ->ConfigBox:
    """ read yaml file and returns
    Args : 
        path_to_yaml (str) :path like input
    Raises:
        ValuError: if yaml file is empty
        e: empty file
        Returns: ConfigBox: ConfigBox type
        """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logging.info(f'yaml file:{path_to_yaml} loaded successfully' )
            return ConfigBox(content)
    except BoxValueError as e:
        raise fraudDetectionException(e,sys) from e
    
def write_yaml(file_path:str,data:dict=None) ->yaml:
    """create yaml file
    Args:
        file _path: str
        data:dict """
    try:
        create_directories(file_path)
        with open(file=file_path,mode='w') as yaml_file:
            if data is not None:
                yaml.dump(data=data,stream=yaml_file)
    except Exception as e:
        raise fraudDetectionException(e,sys) from e
                        
@ensure_annotations
def save_json(path:Path, data:dict):
    """ save json data
    Args: 
        path(Path): path to json file
        data(dict): data to be saved in json format
    """
    try:
        with open(path,'w') as f:
            json.dump(data,f, indent=4)
        logging.info ("f json file saved at: {path}")
    except Exception as e:
        raise fraudDetectionException(e,sys) from e

@ensure_annotations
def load_json(path:Path) ->ConfigBox:
    """load json files data
    Args: 
    path(Path): path to json files 
    """
    try:
        with open(path) as f:
            content=json.load(f)
        logging.info("json file loaded succesfully from:{path}")
        return ConfigBox(content)
    except Exception as e:
        raise fraudDetectionException(e,sys) from e

@ensure_annotations
def get_size(path:Path) ->str:
    """get size in kb
    Args: 
        path(Path): path to file
    Returns: 
        str: size in kb
    """
    size_in_kb = round(os.path.getsize(path)/1024)
    return f"~ {size_in_kb} KB"

@ensure_annotations
def save_numpy_array_data(file_path:Path, array:np.array):
    """ Save numpy array data to file 
    Args:
        file path(Path): location of file to save
        array (numpy array): data to save
    """
    try:
        create_directories(file_path)
        with open(file=file_path,mode='wb') as f:
            np.save(f,array)
    except Exception as e:
        raise fraudDetectionException(e,sys) from e

@ensure_annotations
def load_numpy_array_data(file_path:Path,) -> np.array:
    """load numpy array data from file
    Args:
        file_path(Path): location of file to load
        return(np.array): data loaded
    """
    try:
        with open(file=file_path) as f:
            return np.load(f)
    except Exception as e:
        raise fraudDetectionException(e,sys) from e
    
@ensure_annotations
def save_object(file_path:Path,obj):
    """Save a 
    Args:
        file_path(Path): location to save the object
        obj: Any sort of object
    """
    try:
        create_directories(file_path)
        with open(file=file_path,mode='wb') as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise fraudDetectionException(e,sys) from e
    
@ensure_annotations
def compare_schema(schema: dict, csv_path: Path):
    """
    Compare the schema in a dictionary format with the schema of a CSV file.
    
    Parameters:
        schema (dict): The schema in a dictionary format, where the keys represent column names and 
            the values represent data types.
        csv_path (pathlib.Path): The path to the CSV file.
    
    Raises:
        FraudDetectionException: If the schemas do not match.
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise FraudDetectionException("Failed to read CSV file.", f"{str(e)}")
    
    df_schema = df.dtypes.to_dict()
    
    for col, dtype in schema.items():
        if col not in df_schema:
            raise FraudDetectionException(f"Column '{col}' not found in CSV schema.", "")
        if str(df_schema[col]) != str(dtype):
            raise FraudDetectionException(f"Column '{col}' has a different data type in CSV schema.", 
                                           f"Expected data type: {dtype}, Actual data type: {df_schema[col]}")
