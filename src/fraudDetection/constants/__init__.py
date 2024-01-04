"""
Module: fraudDetection.constants

This module defines constants used in the fraud detection system.

Attributes:
    ROOT_DIR (Path): Root directory of the project.
    CONFIG_DIR (str): Directory containing configuration files.
    CONFIG_FILE_NAME (str): Name of the main configuration file.
    CONFIG_FILE_PATH (Path): Full path to the main configuration file.
    DATA_SCHEMA_FILE_NAME_KEY (str): Key for the data schema file name in the configuration.
    DATA_SCHEMA_DIR (Path): Directory containing data schema files.
    DATA_SCHEMA_COLUMNS_KEY (str): Key for the data schema columns in the configuration.
    DATA_SCHEMA_CATEGORICAL_COLUMN_KEY (str): Key for categorical columns in the data schema.
    DATA_SCHEMA_NUMERICAL_COLUMN_KEY (str): Key for numerical columns in the data schema.
    DATA_SCHEMA_TARGET_COLUMN_KEY (str): Key for the target column in the data schema.
    LOGS_DIR (str): Directory for log files.
    CURRENT_TIME_STAMP (str): Current timestamp.

    TRAINING_PIPELINE_CONFIG_KEY (str): Key for training pipeline configuration.
    TRAINING_PIPELINE_NAME_KEY (str): Key for the name of the training pipeline.
    TRAINING_PIPELINE_ARTIFACT_DIR_KEY (str): Key for the artifacts root directory in the training pipeline.

    DATA_INGESTION_CONFIG_KEY (str): Key for data ingestion configuration.
    DATA_INGESTION_ARTIFACT_DIR_KEY (str): Key for the data ingestion artifacts directory.
    DATA_INGESTION_URL_KEY (str): Key for the source URL in data ingestion configuration.
    DATA_INGESTION_RAW_DATA_DIR_KEY (str): Key for the raw data directory in data ingestion configuration.
    DATA_INGESTION_INGESTED_DIR_KEY (str): Key for the ingested data directory in data ingestion configuration.
    DATA_INGESTION_TRAIN_DIR_KEY (str): Key for the ingested train data directory in data ingestion configuration.
    DATA_INGESTION_TEST_DIR_KEY (str): Key for the ingested test data directory in data ingestion configuration.
    DATA_INGESTION_TEST_SIZE_KEY (str): Key for the test size in data ingestion configuration.
    DATA_INGESTION_STRATIFY_COL_KEY (str): Key for the stratify column in data ingestion configuration.
    DATA_INGESTION_KAGGLE_CONFIG_FILE_PATH (str): Key for the Kaggle configuration file path in data ingestion.

    DATA_VALIDATION_CONFIG_KEY (str): Key for data validation configuration.
    DATA_VALIDATION_ARTIFACT_DIR_KEY (str): Key for data validation artifacts directory.
    DATA_VALIDATION_REPORT_FILE_NAME_KEY (str): Key for the report file name in data validation configuration.
    DATA_VALIDATION_REPORT_FILE_PATH (str): Key for the report file directory in data validation configuration.

    DATA_TRANSFORMATION_CONFIG_KEY (str): Key for data transformation configuration.
    DATA_TRANSFORMATION_ARTIFACTS_DIR_KEY (str): Key for the transformed data directory in data transformation configuration.
    DATA_TRANSFORMED_TRAIN_DIR_KEY (str): Key for the transformed train data directory in data transformation.
    DATA_TRANSFORMED_TEST_DIR_KEY (str): Key for the transformed test data directory in data transformation.
    DATA_TRANSFORMATION_PREPROCESSING_DIR_KEY (str): Key for the preprocessing directory in data transformation.
    DATA_TRANSFORMATION_PREPROCESSING_OBJECT_FILE_NAME_KEY (str): Key for the preprocessing object file name.
    DATA_TRANSFORMATION_IMPUTER_SAMPLER_OBJECT_FILE_NAME_KEY (str): Key for the imputer sampler object file name.

    MODEL_TRAINER_ARTIFACTS_DIR_KEY (str): Key for model trainer artifacts directory.
    MODEL_TRAINER_CONFIG_KEY (str): Key for model trainer configuration.
    MODEL_TRAINED_DIR_KEY (str): Key for the trained model directory.
    MODEL_TRAINED_FILE_NAME_KEY (str): Key for the model file name in model trainer configuration.
    MODEL_TRAINER_CONFIG_DIR (Path): Directory containing model trainer configuration files.
    MODEL_TRAINER_MODEL_CONFIG_FILE_NAME_KEY (str): Key for the model config file name in model trainer configuration.
    MODEL_TRAINER_BASE_SCORE_KEY (str): Key for the base score in model trainer configuration.
    MODEL_TRAINED_DIFF_TRAIN_TEST_ACC_KEY (str): Key for the threshold difference between train and test accuracy.

    MODEL_EVALUATION_CONFIG_KEY (str): Key for model evaluation configuration.
    MODEL_EVALUATION_ARTIFACTS_DIR_KEY (str): Key for model evaluation artifacts directory.
    MODEL_EVALUATION_FILE_NAME_KEY (str): Key for the model evaluation file name.
    BEST_MODEL_KEY (str): Key for the best model in model evaluation .
    HISTORY_KEY (str): Key for best model archive in model evaluation.
    MODEL_PATH_KEY (str): Key for the model path in model evaluation.

    MODEL_PUSHER_CONFIG_KEY (str): Key for model pusher configuration.
    MODEL_PUSHER_EXPORT_DIR_KEY (str): Key for the model export directory.

    FACTORY_GRID_SEARCH_KEY (str): Key for grid search in model factory configuration.
    FACTORY_MODULE_KEY (str): Key for module in model factory configuration.
    FACTORY_CLASS_KEY (str): Key for class in model factory configuration.
    FACTORY_PARAMS_KEY (str): Key for params in model factory configuration.
    FACTORY_MODEL_SELECTION_KEY (str): Key for model selection in model factory configuration.
    FACTORY_SEARCH_PARAM_GRID_KEY (str): Key for search parameter grid in model factory configuration.

    EXPERIMENT_DIR_NAME (str): Directory name for experiments.
    EXPERIMENT_FILE_NAME (str): File name for experiment details.
"""


from datetime import datetime
from pathlib import Path

import os


def get_current_time_stamp():
    """Get the current timestamp.
    Returns:
        str: Current timestamp in a string format.
    """
    return f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"


ROOT_DIR = Path(os.getcwd())
CONFIG_DIR = "configs"
CONFIG_FILE_NAME = "config.yaml"
CONFIG_FILE_PATH: Path = Path(os.path.join(ROOT_DIR, CONFIG_DIR, CONFIG_FILE_NAME))
DATA_SCHEMA_FILE_NAME_KEY = "schema_file_name"
DATA_SCHEMA_DIR = Path(os.path.join(ROOT_DIR, CONFIG_DIR))
DATA_SCHEMA_COLUMNS_KEY = "columns"
DATA_SCHEMA_CATEGORICAL_COLUMN_KEY = "categorical_columns"
DATA_SCHEMA_NUMERICAL_COLUMN_KEY = "numerical_columns"
DATA_SCHEMA_TARGET_COLUMN_KEY = "target_column"
LOGS_DIR = "logs"

CURRENT_TIME_STAMP = get_current_time_stamp()

# Training pipeline related variable
TRAINING_PIPELINE_CONFIG_KEY = "training_pipeline_config"
TRAINING_PIPELINE_NAME_KEY = "pipeline_name"
TRAINING_PIPELINE_ARTIFACT_DIR_KEY = "artifacts_root"

# Data Ingestion related Variable
DATA_INGESTION_CONFIG_KEY = "data_ingestion_config"
DATA_INGESTION_ARTIFACT_DIR_KEY = "data_ingestion"
DATA_INGESTION_URL_KEY = "source_url"
DATA_INGESTION_RAW_DATA_DIR_KEY = "raw_data_dir"
DATA_INGESTION_INGESTED_DIR_KEY = "ingested_dir"
DATA_INGESTION_TRAIN_DIR_KEY = "ingested_train_dir"
DATA_INGESTION_TEST_DIR_KEY = "ingested_test_dir"
DATA_INGESTION_TEST_SIZE_KEY = "test_size"
DATA_INGESTION_STRATIFY_COL_KEY = "stratify"
DATA_INGESTION_KAGGLE_CONFIG_FILE_PATH = "kaggle_config_file"

# Data Validation related variable
DATA_VALIDATION_CONFIG_KEY = "data_validation_config"
DATA_VALIDATION_ARTIFACT_DIR_KEY = "data_validation"
DATA_VALIDATION_REPORT_FILE_NAME_KEY = "report_file_name"
DATA_VALIDATION_REPORT_FILE_PATH = "report_file_dir"

# Data transformation related variable
DATA_TRANSFORMATION_CONFIG_KEY = "data_transformation_config"
DATA_TRANSFORMATION_ARTIFACTS_DIR_KEY = "transformed_dir"
DATA_TRANSFORMED_TRAIN_DIR_KEY = "transformed_train_dir"
DATA_TRANSFORMED_TEST_DIR_KEY = "transformed_test_dir"
DATA_TRANSFORMATION_PREPROCESSING_DIR_KEY = "preprocessing_dir"
DATA_TRANSFORMATION_PREPROCESSING_OBJECT_FILE_NAME_KEY = "preprocessed_object_file_name"
DATA_TRANSFORMATION_IMPUTER_SAMPLER_OBJECT_FILE_NAME_KEY = (
    "imputer_sampler_object_file_name"
)

# Model trainer related variable
MODEL_TRAINER_ARTIFACTS_DIR_KEY = "model_trainer"
MODEL_TRAINER_CONFIG_KEY = "model_trainer_config"
MODEL_TRAINED_DIR_KEY = "trained_model_dir"
MODEL_TRAINED_FILE_NAME_KEY = "model_file_name"
MODEL_TRAINER_CONFIG_DIR = os.path.join(ROOT_DIR, CONFIG_DIR)
MODEL_TRAINER_MODEL_CONFIG_FILE_NAME_KEY = "model_config_file_name"
MODEL_TRAINER_BASE_SCORE_KEY = "base_score"
MODEL_TRAINED_DIFF_TRAIN_TEST_ACC_KEY = "threshold_diff_train_test_acc"

# Model evaluation related variable
MODEL_EVALUATION_CONFIG_KEY = "model_evaluation_config"
MODEL_EVALUATION_ARTIFACTS_DIR_KEY = "model_evaluation_dir"
MODEL_EVALUATION_FILE_NAME_KEY = "model_evaluation_file_name"
BEST_MODEL_KEY = "best_model"
HISTORY_KEY = "history"
MODEL_PATH_KEY = "model_path"

# Model pusher related variable
MODEL_PUSHER_CONFIG_KEY = "model_pusher_config"
MODEL_PUSHER_EXPORT_DIR_KEY = "model_export_dir"

# model factory related variable
FACTORY_GRID_SEARCH_KEY = "grid_search"
FACTORY_MODULE_KEY = "module"
FACTORY_CLASS_KEY = "class"
FACTORY_PARAMS_KEY = "params"
FACTORY_MODEL_SELECTION_KEY = "model_selection"
FACTORY_SEARCH_PARAM_GRID_KEY = "search_param_grid"

EXPERIMENT_DIR_NAME = "experiment"
EXPERIMENT_FILE_NAME = "experiment.csv"
