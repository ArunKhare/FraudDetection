"""
This module provides functionality for creating a custom logger and transforming log messages into a DataFrame.

Functions:
    - get_log_file_name(): Generates a log file name based on the current timestamp.
    - get_log_dataframe(file_path): Transforms log file messages into a Pandas DataFrame.

Constants:
    - LOGS_DIR: Directory path where log files are stored.
    - LOG_FILE_NAME: Name of the log file generated based on the current timestamp.
    - LOG_FILE_PATH: Full path to the log file.

Logger Configuration:
    The logging is configured using the basicConfig method from the logging module. The log file is created
    in the LOGS_DIR directory with a filename based on the current timestamp. The log format includes
    timestamp, log level, line number, file name, function name, and the log message.

Log DataFrame Transformation:
    The get_log_dataframe function reads the log file specified by the file_path parameter and transforms
    the log messages into a Pandas DataFrame. The resulting DataFrame includes columns such as timestamp,
    log level, line number, file name, function name, and the log message.

Examples:
    - log_file_name = get_log_file_name()  # "log_2022-01-01_12-30-45.log"
    - log_dataframe = get_log_dataframe(LOG_FILE_PATH)
"""

import logging
import os
import pandas as pd

from fraudDetection.constants import get_current_time_stamp, LOGS_DIR

def get_log_file_name():
    return f'log_{get_current_time_stamp()}.log'

LOG_FILE_NAME = get_log_file_name()
os.makedirs(LOGS_DIR, exist_ok=True)
LOG_FILE_PATH = os.path.join(LOGS_DIR, LOG_FILE_NAME)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    filemode="w",
    format="[%(asctime)s]^;%(levelname)s^;%(lineno)s^;%(filename)s^;%(funcName)s()^;%(message)s",
    level=logging.INFO,
)


def get_log_dataframe(file_path):
    """Transform log file messages to DataFrame.

    Args:
        file_path (str): Path to the log file.

    Returns:
        pd.DataFrame: DataFrame containing log messages.
    """
    data = []

    with open(file_path) as log_file:
        for line in log_file.readlines():
            data.append(line.split("^;"))

    df = pd.DataFrame(data)
    log_df = df.dropna()

    columns = [
        "Time Stamp",
        "Log Level",
        "Line Number",
        "File Name",
        "Function Name",
        "Message",
    ]
    log_df.columns = columns
    log_df["log_message"] = (
        log_df["Time Stamp"].astype(str, copy=False) + ":$" + log_df["Message"]
    )
    return log_df[["log_message"]]
