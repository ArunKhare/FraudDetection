"""
Module: fraudDetection.utils

This module provides utility functions for handling data, such as reading and writing files, saving and loading objects,
and validating data schema.

Functions:
    create_directories(path_to_directories: list, verbose=True)
    read_yaml(file_path: Path) -> dict
    write_yaml(file_path: Path, data: dict) -> yaml
    write_pickle(file_path: Path, obj: object) -> pickle
    save_json(path: Path, data: dict) -> None
    load_json(path: Path) -> ConfigBox
    get_size(path: Path) -> str
    save_numpy_array_data(file_path: str, array: np.ndarray) -> None
    load_numpy_array_data(file_path: Path) -> np.ndarray
    save_object(file_path: Path, obj) -> None
    load_object(file_path: Path) -> object
    compare_schema(schema: dict, csv_path: Path) -> None
    save_dfs_to_csv(df, file_path: Path, chunk_size: int) -> None
    get_dtypes(df) -> ConfigBox
    load_data(file_path: Path, schema: dict, args: tuple = ()) -> pd.DataFrame
    concat_csv_files(files: list, file_dir: Path) -> pd.DataFrame
    validate_schema(df: pd.DataFrame, schema: dict) -> bool
    is_dir_empty(path: Path) -> bool
    check_data_dir(directory: Path) -> None
"""
import os
import sys
from glob import glob
from pathlib import Path
import dill
import json
import numpy as np
import pandas as pd
from box import ConfigBox
from ensure import ensure_annotations
from tqdm import tqdm
import pandas as pd

# import ruamel.yaml
import yaml
from fraudDetection.exception import FraudDetectionException
from fraudDetection.logger import logging
import pickle


@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """
    Create a list of directories.
    Args:
        path_to_directories (List): List of paths of directories.
        verbose (bool): Whether to print messages about file creation. Default is True.
    """
    try:
        for path in path_to_directories:
            os.makedirs(path, exist_ok=True)
            if not os.path.exists(path):
                open(path, "w").close()
                if verbose:
                    logging.info(f"Created file at: {path}")
            elif verbose:
                logging.info(f"File already exists: {path}")
    except Exception as e:
        raise FraudDetectionException(e, sys) from e

    # @ ensure_annotations


def read_yaml(file_path: Path) -> dict:
    """Read a YAML file and return its content as a dictionary.
    Args:
        file_path (Path): Path to the YAML file.
    Returns:
        dict: Content of the YAML file.
    Raises:
        FraudDetectionException: If an error occurs while reading the file.
    """
    try:
        # yaml = ruamel.yaml.YAML()
        if not file_path.is_file():
            raise FileNotFoundError(f"{file_path} does not exist.")
        if os.stat(file_path).st_size == 0:
            raise ValueError(f"{file_path} is empty.")
        file_path = str(file_path)

        with open(file_path, "r") as f:
            content = yaml.load(f, Loader=yaml.Loader)
            logging.info(f"yaml file:{file_path} loaded successfully")
            return content

    except ValueError as e:
        logging.info(e)
    except FileNotFoundError as e:
        raise FraudDetectionException(e, sys) from e


# @ensure_annotations
def write_yaml(file_path: Path, data: dict) -> yaml:
    """Create a YAML file and write data to it.

    Args:
        file_path (Path): Path to the YAML file.
        data (dict): Data to be written to the YAML file.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file=file_path, mode="w") as yaml_file:
            yaml.dump(data=data, stream=yaml_file)
    except Exception as e:
        raise FraudDetectionException(e, sys) from e


def write_pickle(file_path: Path, obj: object) -> pickle:
    """Serialize an object and save it to a file using pickle.
    Args:
        file_path (Path): Path to the file.
        obj (object): Object to be serialized and saved.
    Raises:
        FraudDetectionException: If an error occurs while saving the object.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        ser_obj = pickle.dumps(obj=obj)
        with open(file=file_path, mode="wb") as file:
            file.write(ser_obj)

    except Exception as e:
        raise FraudDetectionException(e, sys) from e


# @ensure_annotations
def save_json(path: Path, data: dict) -> None:
    """Save data in JSON format to a file.
    Args:
        path (Path): Path to the JSON file.
        data (dict): Data to be saved in JSON format.
    """
    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=4)
        logging.info("f json file saved at: {path}")
    except Exception as e:
        raise FraudDetectionException(e, sys) from e


@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """Load data from a JSON file.
    Args:
        path (Path): Path to the JSON file.
    Returns:
        ConfigBox: Configuration box containing the loaded data.
    Raises:
        FraudDetectionException: If an error occurs while loading the JSON file.
    """
    try:
        with open(path) as f:
            content = json.load(f)
        logging.info("json file loaded successfully from:{path}")
        return ConfigBox(content)
    except Exception as e:
        raise FraudDetectionException(e, sys) from e


@ensure_annotations
def get_directory_size(directory):
    """ "Get size of the directory in MB
    Args:
        directory (Path): directory path
    """
    total_size = 0
    for dirpath, _, filenames in os.walk(directory):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            total_size += os.path.getsize(filepath)
    return round(total_size)


def save_numpy_array_data(file_path: str, array: np.ndarray):
    """Save numpy array data to a file.
    Args:
        file_path (str): Path to the file.
        array (np.ndarray): Numpy array data to be saved.
    Raises:
        FraudDetectionException: If an error occurs while saving the data."""
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(name=dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise FraudDetectionException(e, sys) from e


@ensure_annotations
def load_numpy_array_data(file_path: Path) -> np.ndarray:
    """Load numpy array data from a file.
    Args:
        file_path (Path): Path to the file.
    Returns:
        np.ndarray: Loaded numpy array data.
    Raises:
        FraudDetectionException: If an error occurs while loading the data."""
    try:
        with open(file=file_path, mode="rb") as f:
            return np.load(f)
    except Exception as e:
        raise FraudDetectionException(e, sys) from e


def save_object(file_path: Path, obj):
    """
    Save an object to a file using dill serialization.
    Args:
        file_path (Path): Path to the file.
        obj: Object to be saved.
    Raises:
        FraudDetectionException: If an error occurs while saving the object.
    """

    try:
        dir_path: str = os.path.dirname(p=file_path)
        os.makedirs(name=dir_path, exist_ok=True)
        with open(file=file_path, mode="wb") as file_obj:
            dill.dump(obj=obj, file=file_obj)
    except Exception as e:
        raise FraudDetectionException(e, sys) from e


@ensure_annotations
def load_object(file_path: Path):
    """Load an object from a file using dill deserialization.
    Args:
        file_path (Path): Path to the file.
    Returns:
        object: Loaded object.
    Raises:
        FraudDetectionException: If an error occurs while loading the object.
    """
    try:
        with open(file=file_path, mode="rb") as file_obj:
            return dill.load(file=file_obj)
    except Exception as e:
        raise FraudDetectionException(e, sys) from e


@ensure_annotations
def compare_schema(schema: dict, csv_path: Path) -> None:
    """Compare the schema in a dictionary format with the schema of a CSV file.
    Args:
        schema (dict): Schema in a dictionary format.
        csv_path (Path): Path to the CSV file.
    Raises:
        FraudDetectionException: If the schemas do not match."""
    try:
        df: pd.DataFrame = pd.read_csv(filepath_or_buffer=csv_path)
    except Exception as e:
        raise FraudDetectionException("Failed to read CSV file.", f"{str(e)}")

    df_schema = df.dtypes.to_dict()

    for col, dtype in schema.items():
        if col not in df_schema:
            raise FraudDetectionException(
                f"Column '{col}' not found in CSV schema.", ""
            )
        if str(object=df_schema[col]) != str(object=dtype):
            raise FraudDetectionException(
                f"Column '{col}' has a different data type in CSV schema.",
                f"Expected data type: {dtype}, Actual data type: {df_schema[col]}",
            )


# @ensure_annotations
def save_dfs_to_csv(df, file_path: Path, chunk_size: int) -> None:
    """Save DataFrames to CSV files in chunks.
    Args:
        df: DataFrame to be saved.
        file_path (Path): Path to the CSV file.
        chunk_size (int): Number of elements to write at a time.
    Raises:
        FraudDetectionException: If an error occurs while saving the data.
    """
    try:
        with tqdm(total=len(df), desc=f"Saving test data to CSV {file_path}") as pbar:
            for i, (_, chunk) in enumerate(df.groupby(df.index // chunk_size)):
                chunk_file_path = Path(file_path / f"file_{i}.csv")
                print(f"chunk_file_path: {chunk_file_path}")
                chunk.to_csv(chunk_file_path, index=False)

                pbar.update(n=len(chunk))

    except Exception as e:
        raise FraudDetectionException(e, sys) from e


@ensure_annotations
def get_dtypes(df) -> ConfigBox:
    """Get data types of DataFrame columns.
    Args:
        df: DataFrame.
    Returns:
        ConfigBox: Configuration box containing data types.
    """
    try:
        dtypes_groups = df.columns.to_series().groupby(df.dtypes)

        # create dictionary of dtypes and columns
        dtypes_dict = {}
        for TYPE, col_names in dtypes_groups:
            dtypes_dict[str(TYPE)] = list(col_names)

        # print the dictionary
        logging.info(f"datatypes of {df}")
        return ConfigBox(dtypes_dict)
    except Exception as e:
        raise FraudDetectionException(e, sys) from e


@ensure_annotations
def load_data(file_path: Path, schema: dict, args: tuple = ()) -> pd.DataFrame:
    """Load data from CSV files, validate the schema, and return a DataFrame.
    Args:
        file_path (Path): Path to the directory containing CSV files.
        schema (dict): Schema for the DataFrame.
        args (tuple): Tuple containing start_index and end_index for selecting files.
    Returns:
        pd.DataFrame: Concatenated data from selected CSV files.
    Raises:
        FraudDetectionException: If an error occurs during data loading or schema validation.
    """

    try:
        start_index, end_index = args if len(args) > 0 else (0, 2)

        # Get a list of files from train and test data dir
        dataset_files = glob(pathname=str(object=file_path / "*.csv"))
        if not dataset_files:
            raise Exception(f"No CSV files found in {file_path}")

        if start_index is None:
            start_index = 0
        if end_index is None:
            end_index = len(dataset_files)

        is_file_path_exist = os.path.exists(path=file_path)
        if not is_file_path_exist:
            raise Exception(f"train_file_dir {file_path} does not exist", sys)

        logging.info(msg=f"first train file with first files as : {dataset_files[0]}")

        # concat all  the files in the train and test directory and convert them to dataframe
        dataset_files: list[str] = dataset_files[start_index:end_index]

        num_files: int = len(dataset_files)

        data_df = concat_csv_files(dataset_files, file_path)

        logging.info(f"df and test_df created. Total train_dir CSV files :{num_files}")
        error_message = []
        for column in list(schema.keys()):
            if column in data_df.columns:
                data_df[column].astype(schema[column])
            else:
                error_message.append(f"Column: {column} is not in the schema")

        if len(error_message) > 0:
            raise Exception(error_message)
        column_list = list(schema.keys())
        return data_df[column_list]
    except Exception as e:
        raise FraudDetectionException(e, sys) from e


@ensure_annotations
def concat_csv_files(files: list, file_dir: Path) -> pd.DataFrame:
    """
    Concatenate CSV files into a DataFrame.
    Args:
        files (list): List of file names.
        file_dir (Path): Directory containing the CSV files.
    Returns:
        pd.DataFrame: Concatenated data from CSV files.
    Raises:
        FraudDetectionException: If an error occurs during concatenation.
    """
    try:
        dfs = []
        for i, file in enumerate(iterable=files):
            if file.endswith(".csv"):
                path = Path(os.path.join(file_dir, file))
                df: pd.DataFrame = pd.read_csv(filepath_or_buffer=path)
                dfs.append(df)
        return pd.concat(objs=dfs)
    except Exception as e:
        raise FraudDetectionException(e, sys) from e


@ensure_annotations
def validate_schema(df: pd.DataFrame, schema: dict) -> [True]:
    """
    Validate the schema of a DataFrame against a given schema.
    Args:
        df (pd.DataFrame): DataFrame to be validated.
        schema (dict): Expected schema.
    Raises:
        FraudDetectionException: If the schema validation fails.
    """
    try:
        missing_cols = [col for col in schema if col not in df.columns]
        mismatch_type_cols = [
            col
            for col, dtype in schema.items()
            if col in df.columns and df[col].dtype != dtype
        ]

        if missing_cols:
            raise Exception(
                f"The following columns are missing from the DataFrame schema: {missing_cols}"
            )
        elif mismatch_type_cols:
            raise Exception(
                f"The following columns data types does not match with schema: {mismatch_type_cols}"
            )
        else:
            logging.info(f"train test file schema are as per the Schema {schema}")

        logging.info("Data validated")
        return True
    except Exception as e:
        raise FraudDetectionException(
            error_message=f"Error comparing schema: {e}", error_details=sys
        ) from e


@ensure_annotations
def is_dir_empty(path: Path) -> bool:
    """
    Check if a directory is empty.
    Args:
        path (Path): Path to the directory.
    Returns:
        bool: True if the directory is empty, False otherwise.
    """
    with os.scandir(path) as entries:
        for _ in entries:
            return False
    return True


@ensure_annotations
def check_data_dir(directory: Path) -> None:
    """Check if the raw data directory exists and contains files.
    Args:
        directory (Path): Path to the raw data directory.
    Raises:
        ValueError: If the raw data directory is not found, is not a directory, or is empty.
    """
    directory_str = directory.as_posix()
    if not os.path.exists(path=directory_str):
        raise ValueError("Raw data directory does not exist.")
    if not os.path.isdir(s=directory_str):
        raise ValueError("Raw data directory is not a directory.")
    if not os.listdir(path=directory_str):
        raise ValueError(f"No files found in the raw data directory: {directory_str}")
    if not os.listdir(path=directory_str)[0]:
        raise ValueError("Raw data directory is empty.")
