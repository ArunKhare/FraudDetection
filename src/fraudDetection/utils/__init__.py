import os, sys
import numpy as np
import pandas as pd
import dill, yaml, json
from box import ConfigBox
from pathlib import Path
from ensure import ensure_annotations
from box.exceptions import BoxValueError

from fraudDetection.exception import FraudDetectionException
from fraudDetection.logger import logging


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
        raise FraudDetectionException(e, sys) from e 
    
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
        raise FileNotFoundError(f"Config file not found:{path_to_yaml}")
    except FileNotFoundError as e:
        raise FraudDetectionException(e,sys) from e
    except BoxValueError as e:
        raise FraudDetectionException(e,sys) from e
    
@ensure_annotations    
def write_yaml(file_path:Path,data:dict=None) ->yaml:
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
        raise FraudDetectionException(e,sys) from e
                        
@ensure_annotations
def save_json(path:Path, data:dict)  -> None:
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
        raise FraudDetectionException(e,sys) from e

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
        raise FraudDetectionException(e,sys) from e

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


# def save_numpy_array_data(file_path, array, chunk_size= 1024):
#     """ Save numpy array data to file in chunks
#     Args:
#         file_path : str location of file to save
#         array : numpy array data to save
#         chunk_size : number of elements to write at a time, defaults to 1024
#     """
#     try:
#         create_directories([file_path])
#         with open(file=file_path, mode='a') as f:
#             for i in range(0, len(array), chunk_size):
#                 np.save(f, array[i:i+chunk_size])
#                 f.flush()
#                 os.fsync(f.fileno())
#             np.save(f,array)
#     except Exception as e:
#         raise FraudDetectionException(e,sys) from e

def save_numpy_array_data(file_path: str, array: np.array):
    """
    Save numpy array data to file
    file_path: str location of file to save
    array: np.array data to save
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise FraudDetectionException(e, sys) from e
    
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
        raise FraudDetectionException(e,sys) from e
    
# @ensure_annotations
# def save_object(file_path:Path,obj) -> None:
#     """Save a 
#     Args:
#         file_path(Path): location to save the object
#         obj: Any sort of object
#     """
#     try:
#         create_directories([file_path])
#         with open(file=file_path,mode='wb') as file_obj:
#             dill.dump(obj,file_obj)
#     except Exception as e:
#         raise FraudDetectionException(e,sys) from e
    
def save_object(file_path:str,obj):
    """
    file_path: str
    obj: Any sort of object
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise FraudDetectionException(e,sys) from e

@ensure_annotations    
def load_object(file_path:str):
    """
    file_path:str
    """
    try:
        with open(file=file_path,mode='rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise FraudDetectionException(e,sys) from e

@ensure_annotations
def compare_schema(schema: dict, csv_path: Path) -> None:
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

@ensure_annotations
def save_dfs_to_csv(df, file_path: Path, chunk_size:int ) -> None:
    try:
        for i, (index,chunk) in enumerate(df.groupby(df.index // chunk_size)):
            chunk_file_path: Path = file_path / f"file_{i}.csv"
            # chunk.to_csv(chunk_file_path, index=False, header=i == 0)
            chunk.to_csv(chunk_file_path,index=False)
    except Exception as e:
        raise FraudDetectionException(e, sys) from e
    
@ensure_annotations
def get_dtypes(df) -> ConfigBox:
    try:
        dtypes_groups = df.columns.to_series().groupby(df.dtypes)

        # create dictionary of dtypes and columns
        dtypes_dict = {}
        for dtype, col_names in dtypes_groups:
            dtypes_dict[str(dtype)] = list(col_names)

        # print the dictionary
        logging.info(f'datatypes of {df}')
        return dtypes_dict
    except Exception as e:
        raise FraudDetectionException(e,sys) from e

@ensure_annotations
def load_data(file_path:Path,schema:dict) ->pd.DataFrame:
        """ validation the schema of data downloaded with the provided schema
        """
        try:
            # logging checking training and test files are available
            is_file_path_exist = os.path.exists(file_path)
            
            if not is_file_path_exist:
                raise Exception(f"train_file_dir {file_path} does not exist", sys)
            
            if not os.listdir(file_path):
                raise Exception(f"{file_path} no file found", sys)
            
            # Get a list of files from train and test data dir
            dataset_files = os.listdir(file_path)
            
            logging.info(f"Is train file with first files as : {dataset_files[0]}")

            # concate all  the files in the train and test directory and convert them to dataframe
            dataset_files: list[str] = dataset_files[:2]
            
            #number of files in each test and train dir
            num_files = len([f for f in dataset_files if os.path.isfile(Path(os.path.join(file_path, f))) if f.endswith('.csv')])
            
            data_df = concat_csv_files(dataset_files,file_path)
            
            logging.info(f"df and test_df created. Total train_dir CSV files :{num_files}")

            error_message = ""
            for column in list(schema.keys()):
                if column in data_df.columns:
                    data_df[column].astype(schema[column])
                else:
                    error_message = f'Column: {column} is not in the schema'

            if len(error_message) > 0:
                raise Exception(error_message)

            return data_df
        except Exception as e:
            raise FraudDetectionException(e,sys) from e

@ensure_annotations  
def concat_csv_files(files: list, file_dir: Path) -> pd.DataFrame:
    try:
        dfs =[]    
        for i, file in enumerate(files):
            if file.endswith('.csv'):
                path = Path(os.path.join(file_dir,file))
                df = pd.read_csv(path)
                dfs.append(df)
        return pd.concat(dfs)
    except Exception as e:
        raise FraudDetectionException(e,sys) from e
    
@ensure_annotations    
def validate_schema(df:pd.DataFrame,schema:dict):
    try:
        missing_cols = [col for col in schema if col not in df.columns]
        mismatch_dtype_cols = [col for col, dtype in schema.items() if col in df.columns and df[col].dtype != dtype]

        if missing_cols:
            raise Exception (f"The following columns are missing from the DataFrame schema: {missing_cols}")
        elif mismatch_dtype_cols:
            raise Exception (f"The following columns data types does not match with schema: {mismatch_dtype_cols}")
        else:
            logging.info(f"train test file schema are as per the Schema {schema}")
        
        logging.info("Data validated")
        return True
    except Exception as e:
        raise FraudDetectionException(f"Error comparing schema: {e}",sys) from e
