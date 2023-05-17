import os, sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
import dill, yaml, json
from box import ConfigBox
from pathlib import Path ,WindowsPath
from ensure import ensure_annotations
from box.exceptions import BoxValueError
from glob import glob
import yaml
from fraudDetection.exception import FraudDetectionException
from fraudDetection.logger import logging
from tqdm import tqdm
import ruamel.yaml

# Your code using ruamel.yaml

@ensure_annotations 
def create_directories(path_to_directories:list,verbose=True):
    """ create list of directories
    Args: 
        path_to_directories(list): list of path of directories
        ignore_log (bool,optional): ignore if multiple dirs is to be created. Defaults to False.
    """
    try:
         for path in path_to_directories:
            os.makedirs(path, exist_ok=True)
            if not os.path.exists(path):
                open(path, 'w').close()
                if verbose:
                    logging.info(f"Created file at: {path}")
            elif verbose:
                logging.info(f"File already exists: {path}")
    except Exception as e:
        raise FraudDetectionException(e, sys) from e 

@dataclass
class ConfigBox:
    data: str

def construct_Configbox(loader, node):
    data = ConfigBox(loader.construct_mapping(node))
    # Create an instance of the ConfigBox object
    config_box = ConfigBox(data)
    return config_box

# @ ensure_annotations
def read_yaml(file_path:Path) -> dict:
    """ read yaml file and returns
    Args : 
        file_path (str) :path like input
    Raises:
        ValuError: if yaml file is empty
        e: empty file
        Returns: dict
        """
    try:
        yaml = ruamel.yaml.YAML()
        if not file_path.is_file():
            raise FileNotFoundError(f"{file_path} does not exist.")
        if os.stat(file_path).st_size == 0:
            raise ValueError(f"{file_path} is empty.")
        file_path = str(file_path)
  
        with open(file_path, 'r') as f:
            content = yaml.load(f)
            logging.info(f'yaml file:{file_path} loaded successfully' )
            return content
        
    except ValueError as e:
        logging.info(e)
    except FileNotFoundError as e:
        raise FraudDetectionException(e,sys) from e
    
# @ensure_annotations    
def write_yaml(file_path:Path, data: dict) ->yaml:
    """create yaml file
    Args:
        file _path: str
        data:dict """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file=file_path, mode='w') as yaml_file:
            model = yaml.dump(data=data, stream=yaml_file, indent=4, width=80)
            return model
    except yaml.YAMLError as e:
        raise FraudDetectionException(e,sys) from e
    except yaml.error as e:
        raise FraudDetectionException(e,sys) from e
    except Exception as e:
        raise FraudDetectionException(e,sys) from e
         
# @ensure_annotations
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

def save_numpy_array_data(file_path: str, array: np.ndarray):
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
def load_numpy_array_data(file_path:Path) -> np.ndarray:
    """load numpy array data from file
    Args:
        file_path(Path): location of file to load
        return(np.array): data loaded
    """
    try:
        with open(file=file_path, mode='rb') as f:
            return np.load(f)
    except Exception as e:
        raise FraudDetectionException(e,sys) from e
    
def save_object(file_path:Path,obj):
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
def load_object(file_path:Path):
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

# @ensure_annotations
def save_dfs_to_csv(df, file_path: Path, chunk_size:int ) -> None:

    try:
       
        with tqdm(total=len(df), desc=f"Saving test data to CSV {file_path}") as pbar:
           
            for i, ( _, chunk ) in enumerate(df.groupby(df.index // chunk_size)):
                chunk_file_path = Path(file_path / f"file_{i}.csv")
                print(f'chunk_file_path: {chunk_file_path}')
                chunk.to_csv(chunk_file_path, index=False)
      
                pbar.update(len(chunk))
                
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
def load_data(file_path: Path, schema: dict, args : tuple =()) ->pd.DataFrame:
    """Validates the schema of data downloaded with the provided schema.

    Args:
        file_path: The path to the directory containing CSV files.
        schema: The YAML file for a DataFrame.
        start_index: The start index to select the files from a list of files in the directory. Default or 'None' is 0. 
        end_index: The end index to select the files from a list of files in the directory. Default is 2. If `None`, all files in the directory will be selected.

    Returns:
        A DataFrame containing the concatenated data from the selected CSV files.
    """
    
    try:
        start_index, end_index = args if len(args) > 0 else (0, 2)
 
        if start_index is None:
            start_index = 0
        if end_index is None:
            end_index = len(dataset_files)

        is_file_path_exist = os.path.exists(file_path)        
        if not is_file_path_exist:
            raise Exception(f"train_file_dir {file_path} does not exist", sys)

        # Get a list of files from train and test data dir
        dataset_files = glob(str(file_path / "*.csv"))
        if not dataset_files:
            raise Exception(f"No CSV files found in {file_path}")

        logging.info(f"first train file with first files as : {dataset_files[0]}")
        
        # concate all  the files in the train and test directory and convert them to dataframe
        dataset_files: list[str] = dataset_files[start_index:end_index]
      
        num_files = len(dataset_files)
        
        data_df = concat_csv_files(dataset_files,file_path)
        
        logging.info(f"df and test_df created. Total train_dir CSV files :{num_files}")
        error_message = []
        for column in list(schema.keys()):
            if column in data_df.columns:
                data_df[column].astype(schema[column])
            else:
                error_message.append (f'Column: {column} is not in the schema')
      
        if len(error_message) > 0:
            raise Exception(error_message)
        column_list = list(schema.keys())
        return data_df[column_list]
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
    
@ensure_annotations
def is_dir_empty(path: Path) -> bool:
    with os.scandir(path) as entries:
        for entry in entries:
            return False
    return True

@ensure_annotations
def check_data_dir(raw_data_dir: Path) -> None:
    raw_data_dir_str =  raw_data_dir.as_posix()
    if not os.path.exists(raw_data_dir_str):
        raise ValueError("Raw data directory does not exist.")
    if not os.path.isdir(raw_data_dir_str):
        raise ValueError("Raw data directory is not a directory.")
    if not os.listdir(raw_data_dir_str):
        raise ValueError(f"No files found in the raw data directory: {raw_data_dir_str}")
    if not os.listdir(raw_data_dir_str)[0]:
        raise ValueError("Raw data directory is empty.")
