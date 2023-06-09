import sys

from fraudDetection.components import DataIngestion, DataValidation, DataTransformation, ModelTrainer, ModelEvaluation, \
    ModelPusher
from fraudDetection.config.configuration import ConfigurationManager
from fraudDetection.constants import CONFIG_FILE_PATH
from fraudDetection.exception import FraudDetectionException
from fraudDetection.logger import logging


def main():
    try:
        config = ConfigurationManager(CONFIG_FILE_PATH)
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion_artifacts = DataIngestion(data_ingestion_config).initiate_data_ingestion()

        data_validation_config = config.get_data_validation_config()
        data_validation_artifacts = DataValidation(data_ingestion_artifacts,
                                                   data_validation_config).initiate_data_validation()

        data_transformation_config = config.get_data_transformation_config()
        data_transformation_artifacts = DataTransformation(data_validation_artifacts, data_ingestion_artifacts,
                                                           data_transformation_config).initiate_data_transformation()

        model_trainer_config = config.get_model_trainer_config()
        model_trainer_artifacts = ModelTrainer(model_trainer_config,
                                               data_transformation_artifacts).initiate_model_trainer()

        model_evaluation_config = config.get_model_evaluation_config()
        mode_evaluation_artifacts = ModelEvaluation(model_evaluation_config, data_ingestion_artifacts,
                                                    data_validation_artifacts, model_trainer_artifacts,
                                                    data_transformation_artifacts).initiate_model_evaluation()

        model_pusher_config = config.get_model_pusher_config()
        model_pusher_artifacts = ModelPusher(model_pusher_config=model_pusher_config,
                                             model_evaluation_artifact=mode_evaluation_artifacts).initiate_model_pusher()

    except Exception as e:
        logging.error(e)
        raise FraudDetectionException(e, sys) from e


if __name__ == "__main__":
    main()
