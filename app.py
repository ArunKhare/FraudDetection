from fraudDetection.pipeline.pipeline import Pipeline
from fraudDetection.config.configuration import ConfigurationManager, ROOT_DIR, CONFIG_FILE_PATH

import os

config = ConfigurationManager(config=CONFIG_FILE_PATH)
training_pipeline_config = config.get_training_pipeline_config()
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


