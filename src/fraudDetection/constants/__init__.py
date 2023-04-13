from datetime import datetime

import os

def get_current_time_stamp():
    return f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"

ROOT_DIR = os.getcwd()
CONFIG_DIR = "config"
CONFIG_FILE_NAME = "config.yaml"
CONFIG_FILE_PATH = os.path.join(ROOT_DIR,CONFIG_DIR,CONFIG_FILE_NAME)

LOGS_DIR ="logs"

CURRENT_TIME_STAMP = get_current_time_stamp()

DATASET_SCHEMA_COUMNS_KEY = "columns"
# Training pipeline related variable

