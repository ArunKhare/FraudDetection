import logging
from datetime import datetime
import os
import pandas as pd

from fraudDetection.constants import get_current_time_stamp, LOGS_DIR

def get_log_file_name():
    return f"log_{get_current_time_stamp()}.log"

LOG_FILE_NAME = get_log_file_name()
os.makedirs(LOGS_DIR, exist_ok= True)
LOG_FILE_PATH = os.path.join(LOGS_DIR,LOG_FILE_NAME)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    filemode="w",
    format='[%(asctime)s]^;%(levelname)s^;%(lineno)s^;%(filename)s^;%(funcname)s()^;%(message)s',level=logging.INFO
    )

def get_log_dateframe(file_path):
    data = []
    with open(file_path) as log_file:
        for line in log_file.readlines():
            data.append(line.split("^;"))
    log_df = pd.DataFrame(data)
    columns = ["Time Stamp","Log Level","line number","file name","function name","message" ]
    log_df.columns = columns
    log_df["log_message"] = log_df['Time Stamp'].astype(str) +":$"+log_df["message"]
    return log_df[["log_message"]]



