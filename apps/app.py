""""
This module sets up and runs the training pipeline for the Fraud Detection System. It initializes the necessary configurations, directories, and components required for training models. The main functionality is executed by creating an instance of the Pipeline class and calling its run method.

Module Components:
    - Pipeline: Class responsible for orchestrating the training pipeline.
    - ConfigurationManager: Class for managing configuration settings.
    - ROOT_DIR: Root directory of the Fraud Detection System.
    - CONFIG_FILE_PATH: Path to the configuration file.
    - get_current_time_stamp: Function to get the current timestamp.

Configuration and Directory Setup:
    - Initializes the ConfigurationManager with the specified configuration file path.
    - Retrieves training pipeline, model pusher, and model trainer configurations.
    - Defines constants for artifact directory, project folder name, logs directory, model configuration file path, model directory, saved models directory, and pipeline directory.
"""

import os
from fraudDetection.pipeline.pipeline import Pipeline
from fraudDetection.config.configuration import (
    ConfigurationManager,
    ROOT_DIR,
    CONFIG_FILE_PATH,
)

config = ConfigurationManager(config=CONFIG_FILE_PATH)
training_pipeline_config = config.get_training_pipeline_config
model_pusher_config = config.get_model_pusher_config()
model_trainer_config = config.get_model_trainer_config()

ARTIFACT_DIR = training_pipeline_config.artifacts_root
PROJECT_FOLDER_NAME = training_pipeline_config.training_pipeline_name
LOGS_DIR = os.path.join(ROOT_DIR, "logs")
MODEL_CONFIG_FILE_PATH = model_trainer_config.model_config_file_path
MODEL_DIR = model_pusher_config.model_export_dir
SAVED_MODELS_DIRECTORY = model_pusher_config.saved_models_directory
PIPELINE_DIR = os.path.join(ROOT_DIR, "src", "fraudDetection")

pipeline = Pipeline(config=config)

if __name__ == "__main__":
    pipeline.run()
