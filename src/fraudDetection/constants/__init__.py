from datetime import datetime
from pathlib import Path

import os

def get_current_time_stamp():
    return f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"

ROOT_DIR = os.getcwd()
CONFIG_DIR = "configs"
CONFIG_FILE_NAME = "config.yaml"
CONFIG_FILE_PATH: Path = Path(os.path.join(ROOT_DIR,CONFIG_DIR,CONFIG_FILE_NAME))
# CONFIG_FILE_PATH: Path = Path("configs/config.yaml")

LOGS_DIR ="logs"

CURRENT_TIME_STAMP = get_current_time_stamp()

# Training pipeline related variable
TRAINING_PIPELINE_CONFIG_KEY = "training_pipeline_config"
TRAINING_PIPELINE_NAME_KEY = "pipeline_name"
TRAINING_PIPELINE_ARTIFACT_DIR_KEY = "artifacts_root"

# Data Ingestion related Variable
DATA_INGESTION_CONFIG_KEY = "data_ingestion_config"
DATA_INGESTION_ARTIFACT_DIR_KEY ="data_ingestion"
DATA_INGESTION_URL_KEY = "source_url"
DATA_INGESTION_RAW_DATA_DIR_KEY = "raw_data_dir"
DATA_INGESTION_ZIP_DIR_KEY = "zip_dir"
DATA_INGESTION_INGESTED_DIR_KEY = "ingested_dir"
DATA_INGESTION_TRAIN_DIR_KEY = "ingested_train_dir"
DATA_INGESTION_TEST_DIR_KEY = "ingested_test_dir"
DATA_INGESTION_TEST_SIZE_KEY = "test_size"
DATA_INGESTION_STRATIFY_COL_KEY = "stratify"

DATASET_SCHEMA_COUMNS_KEY = "columns"

#Data Validation related variable
DATA_VALIDATION_CONFIG_KEY = "data_validation_config"
DATA_VALIDATION_ARTIFACT_DIR_KEY = "data_validation"
DATA_VALIDATION_SCHEMA_DIR_KEY = "schema_dir"
DATA_VALIDATION_SCHEMA_FILE_NAME_KEY = "Schema_file_name"
DATA_VALIDATION_REPORT_FILE_NAME_KEY = "report.json"

#Data transformation related variable
DATA_TRANSFORMATION_CONFIG_KEY = "data_transformation_config"
DATA_TRANSFORMATION_ARTIFACTS_DIR_KEY = "transformed_dir"
DATA_TRANSFORMED_TRAIN_DIR_KEY = "transformed_train_dir"
DATA_TRANSFORMED_TEST_DIR_KEY = "transformed_test_dir"
DATA_TRANSFORMATION_PREPROCESSING_DIR_KEY = "preprocessing_dir"
DATA_PREPROCESSING_OBJECT_FILE_NAME_KEY = "preprocessed.pkl"

# Model trainer related variable
MODEL_TRAINER_ARTIFACTS_DIR_KEY = "model_trainer"
MODEL_TRAINER_CONFIG_KEY = "model_trainer_config"
MODEL_TRAINED_DIR_KEY = "trained_model_dir"
MODEL_TRAINED_FILE_NAME_KEY = "model_file_name"
MODEL_TRAINER_MODEL_CONFIG_DIR_KEY = "model_config_dir"
MODEL_TRAINER_MODEL_CONFIG_FILE_NAME_KEY = "model_config_file_name"
MODEL_TRAINED_BASE_ACCURACY_KEY = "base_accuracy"

#Model evaluation related variable
MODEL_EVALUATION_CONFIG_KEY = "model_evaluation_config"
MODEL_EVALUATION_ARTIFACTS_DIR_KEY = "model_evaluation"
MODEL_EVALUATION_FILE_NAME_KEY = "model_evaluation.yaml"
MLFLOW_URI_KEY = "mlflow_uri"

#Model pusher related variable
MODEL_PUSHER_CONFIG_KEY = "model_pusher_config"
MODEL_PUSHER_EXPORT_DIR_KEY = "model_export_dir"