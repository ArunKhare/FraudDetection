"""Configuration Manager for Fraud Detection Project.

This module provides a ConfigurationManager class responsible for reading and retrieving
various configurations required for the Fraud Detection project, including data ingestion,
model training, and evaluation.

Classes:
    - ConfigurationManager: Manages the configuration settings for the Fraud Detection project.

Functions:
    - get_current_time_stamp(): Helper function to get the current timestamp.
    - read_yaml(config_path: Path) -> ConfigBox: Helper function to read YAML configuration files.
"""

import sys
from box import ConfigBox

from fraudDetection.entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelEvaluationConfig,
    ModelPusherConfig,
    ModelTrainerConfig,
    TrainingPipelineConfig,
)

from fraudDetection.utils import read_yaml
from fraudDetection.logger import logging
from fraudDetection.constants import *
from fraudDetection.exception import FraudDetectionException


class ConfigurationManager:
    """Configuration Manager for Fraud Detection Project.

    This class is responsible for reading and retrieving various configurations required for
    the Fraud Detection project, including data ingestion, model training, and evaluation.

    Attributes:
        - config (ConfigBox): Parsed configuration settings.
        - current_time_stamp (str): Current timestamp.
        - training_pipeline_config (TrainingPipelineConfig): Configuration for the training pipeline.
        - artifact_dir (Path): Directory for storing project artifacts.

    Methods:
        - __init__(config: Path = CONFIG_FILE_PATH) -> None: Initializes the ConfigurationManager.
        - get_training_pipeline_config() -> TrainingPipelineConfig: Retrieves training pipeline configuration.
        - get_data_ingestion_config() -> DataIngestionConfig: Retrieves data ingestion configuration.
        - get_data_validation_config() -> DataValidationConfig: Retrieves data validation configuration.
        - get_data_transformation_config() -> DataTransformationConfig: Retrieves data transformation configuration.
        - get_model_trainer_config() -> ModelTrainerConfig: Retrieves model trainer configuration.
        - get_model_evaluation_config() -> ModelEvaluationConfig: Retrieves model evaluation configuration.
        - get_model_pusher_config() -> ModelPusherConfig: Retrieves model pusher configuration.
    """

    def __init__(self, config: Path = CONFIG_FILE_PATH) -> None:
        """Initialize the ConfigurationManager.
        Args:
            config (Path): Path to the configuration file (default is CONFIG_FILE_PATH).
        Raises:
            FraudDetectionException: If an error occurs during initialization.
        """
        try:
            self.config: ConfigBox = read_yaml(config)
            self.current_time_stamp: str = get_current_time_stamp()
            self.training_pipeline_config: TrainingPipelineConfig = (
                self.get_training_pipeline_config
            )
            self.artifact_dir: Path = self.training_pipeline_config.artifacts_root
        except Exception as e:
            raise FraudDetectionException(e, sys) from e

    @property
    def get_training_pipeline_config(self) -> TrainingPipelineConfig:
        """Get the TrainingPipelineConfig.
        Returns:
            training_pipeline_config(obj:'TrainingPipelineConfig'): Configuration for the training pipeline.
        Raises:
            FraudDetectionException: If an error occurs while retrieving the configuration.
        """
        try:
            training_pipeline: ConfigBox = self.config[TRAINING_PIPELINE_CONFIG_KEY]
            artifact_dir = Path(
                os.path.join(
                    ROOT_DIR, training_pipeline[TRAINING_PIPELINE_ARTIFACT_DIR_KEY]
                )
            )
            training_pipeline_name = training_pipeline[TRAINING_PIPELINE_NAME_KEY]

            training_pipeline_config = TrainingPipelineConfig(
                artifacts_root=artifact_dir,
                training_pipeline_name=training_pipeline_name,
            )

        except Exception as e:
            raise FraudDetectionException(e, sys) from e
        return training_pipeline_config

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """Get the DataIngestionConfig.
        Returns:
            data_ingestion_config (obj:'DataIngestionConfig'): Configuration for data ingestion.
        Raises:
            FraudDetectionException: If an error occurs while retrieving the configuration.
        """
        try:
            data_ingestion_artifact_dir = Path(
                os.path.join(self.artifact_dir, DATA_INGESTION_ARTIFACT_DIR_KEY)
            )
            data_ingestion_info: ConfigBox = self.config[DATA_INGESTION_CONFIG_KEY]
            dataset_download_url: ConfigBox = data_ingestion_info[
                DATA_INGESTION_URL_KEY
            ]
            kaggle_config_file = Path(
                data_ingestion_info[DATA_INGESTION_KAGGLE_CONFIG_FILE_PATH]
            )
            raw_data_dir = Path(
                os.path.join(
                    data_ingestion_artifact_dir,
                    data_ingestion_info[DATA_INGESTION_RAW_DATA_DIR_KEY],
                )
            )
            ingested_data_dir = Path(
                os.path.join(
                    data_ingestion_artifact_dir,
                    data_ingestion_info[DATA_INGESTION_INGESTED_DIR_KEY],
                )
            )
            ingested_train_dir = Path(
                os.path.join(
                    data_ingestion_artifact_dir,
                    data_ingestion_info[DATA_INGESTION_TRAIN_DIR_KEY],
                )
            )
            ingested_test_dir = Path(
                os.path.join(
                    data_ingestion_artifact_dir,
                    data_ingestion_info[DATA_INGESTION_TEST_DIR_KEY],
                )
            )
            stratify: str = data_ingestion_info[DATA_INGESTION_STRATIFY_COL_KEY]
            test_size: float = data_ingestion_info[DATA_INGESTION_TEST_SIZE_KEY]

            data_ingestion_config = DataIngestionConfig(
                source_url=dataset_download_url,
                kaggle_config_file=kaggle_config_file,
                raw_data_dir=raw_data_dir,
                ingested_dir=ingested_data_dir,
                ingested_train_dir=ingested_train_dir,
                ingested_test_dir=ingested_test_dir,
                stratify=stratify,
                test_size=test_size,
            )

            logging.info(f"Data ingestion config : {data_ingestion_config}")
        except Exception as e:
            raise FraudDetectionException(e, sys) from e

        return data_ingestion_config

    def get_data_validation_config(self) -> DataValidationConfig:
        """Get the DataValidationConfig.
        Returns:
            data_validation_config (obj:'DataValidationConfig'): Configuration for data validation.
        Raises:
            FraudDetectionException: If an error occurs while retrieving the configuration.
        """
        try:
            data_validation_artifacts_dir = Path(
                os.path.join(
                    self.artifact_dir,
                    DATA_VALIDATION_ARTIFACT_DIR_KEY,
                    self.current_time_stamp,
                )
            )
            data_validation_config_info: ConfigBox = self.config[
                DATA_VALIDATION_CONFIG_KEY
            ]
            schema_file_name: str = data_validation_config_info[
                DATA_SCHEMA_FILE_NAME_KEY
            ]
            schema_file_path: Path = Path(
                os.path.join(
                    DATA_SCHEMA_DIR,
                    data_validation_config_info[DATA_SCHEMA_FILE_NAME_KEY],
                )
            )
            report_file_name: str = data_validation_config_info[
                DATA_VALIDATION_REPORT_FILE_NAME_KEY
            ]
            report_file_path = Path(
                os.path.join(data_validation_artifacts_dir, report_file_name)
            )

            data_validation_config = DataValidationConfig(
                schema_file_name=schema_file_name,
                schema_file_path=schema_file_path,
                report_file_name=report_file_name,
                report_file_path=report_file_path,
            )

            logging.info(f"logging data validation config: {data_validation_config}")
        except Exception as e:
            raise FraudDetectionException(e, sys) from e
        return data_validation_config

    def get_data_transformation_config(self) -> DataTransformationConfig:
        """Get the DataTransformationConfig.
        Returns:
            data_transformation_config (obj:'DataTransformationConfig'): Configuration for data transformation.
        Raises:
            FraudDetectionException: If an error occurs while retrieving the configuration.
        """
        try:
            data_transformation_config_info: ConfigBox = self.config[
                DATA_TRANSFORMATION_CONFIG_KEY
            ]
            data_transformation_artifacts_dir = Path(
                os.path.join(
                    self.artifact_dir,
                    data_transformation_config_info[
                        DATA_TRANSFORMATION_ARTIFACTS_DIR_KEY
                    ],
                )
            )
            transformed_train_dir = Path(
                os.path.join(
                    data_transformation_artifacts_dir,
                    data_transformation_config_info[DATA_TRANSFORMED_TRAIN_DIR_KEY],
                )
            )
            transformed_test_dir = Path(
                os.path.join(
                    data_transformation_artifacts_dir,
                    data_transformation_config_info[DATA_TRANSFORMED_TEST_DIR_KEY],
                )
            )
            preprocessing_object_file_path = Path(
                os.path.join(
                    data_transformation_artifacts_dir,
                    data_transformation_config_info[
                        DATA_TRANSFORMATION_PREPROCESSING_DIR_KEY
                    ],
                    data_transformation_config_info[
                        DATA_TRANSFORMATION_PREPROCESSING_OBJECT_FILE_NAME_KEY
                    ],
                )
            )
            imputer_sampler_object_file_path = Path(
                os.path.join(
                    data_transformation_artifacts_dir,
                    data_transformation_config_info[
                        DATA_TRANSFORMATION_PREPROCESSING_DIR_KEY
                    ],
                    data_transformation_config_info[
                        DATA_TRANSFORMATION_IMPUTER_SAMPLER_OBJECT_FILE_NAME_KEY
                    ],
                )
            )

            data_transformation_config = DataTransformationConfig(
                transformed_dir=data_transformation_artifacts_dir,
                transformed_train_dir=transformed_train_dir,
                transformed_test_dir=transformed_test_dir,
                preprocessing_object_file_path=preprocessing_object_file_path,
                imputer_sampler_object_file_path=imputer_sampler_object_file_path,
            )
        except Exception as e:
            raise FraudDetectionException(e, sys) from e
        logging.info(f"Data transformation config: {data_transformation_config}")
        return data_transformation_config

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        """Get the ModelTrainerConfig.
        Returns:
            ModelTrainerConfig: Configuration for the model trainer.
        Raises:
            FraudDetectionException: If an error occurs while retrieving the configuration.
        """
        try:
            model_trainer_artifacts_dir = Path(
                os.path.join(
                    self.artifact_dir,
                    MODEL_TRAINER_ARTIFACTS_DIR_KEY,
                    self.current_time_stamp,
                )
            )

            model_trainer_config_info: ConfigBox = self.config[MODEL_TRAINER_CONFIG_KEY]

            base_accuracy: ConfigBox = model_trainer_config_info[
                MODEL_TRAINER_BASE_SCORE_KEY
            ]
            trained_model_file_path = Path(
                os.path.join(
                    model_trainer_artifacts_dir,
                    model_trainer_config_info[MODEL_TRAINED_DIR_KEY],
                    model_trainer_config_info[MODEL_TRAINED_FILE_NAME_KEY],
                )
            )
            model_config_file_path = Path(
                os.path.join(
                    MODEL_TRAINER_CONFIG_DIR,
                    model_trainer_config_info[MODEL_TRAINER_MODEL_CONFIG_FILE_NAME_KEY],
                )
            )
            threshold_diff_train_test_acc = model_trainer_config_info[
                MODEL_TRAINED_DIFF_TRAIN_TEST_ACC_KEY
            ]

            model_trainer_config = ModelTrainerConfig(
                trained_model_file_path=trained_model_file_path,
                base_score=base_accuracy,
                model_config_file_path=model_config_file_path,
                threshold_diff_train_test_acc=threshold_diff_train_test_acc,
            )
        except Exception as e:
            raise FraudDetectionException(e, sys) from e
        logging.info(f"Model trainer config : {model_trainer_config}")

        return model_trainer_config

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        """Get the ModelEvaluationConfig.
        Returns:
            ModelEvaluationConfig: Configuration for model evaluation.
        Raises:
            response (obj:'FraudDetectionException'): If an error occurs while retrieving the configuration.
        """
        try:
            model_evaluation_config_info: ConfigBox = self.config[
                MODEL_EVALUATION_CONFIG_KEY
            ]
            artifact_dir: Path = Path(
                os.path.join(
                    self.artifact_dir,
                    model_evaluation_config_info[MODEL_EVALUATION_ARTIFACTS_DIR_KEY],
                )
            )

            model_evaluation_file_path = Path(
                os.path.join(
                    artifact_dir,
                    model_evaluation_config_info[MODEL_EVALUATION_FILE_NAME_KEY],
                )
            )

            response = ModelEvaluationConfig(
                model_evaluation_file_path=model_evaluation_file_path,
                time_stamp=self.current_time_stamp,
            )

            logging.info(f"Model Evaluation config: {response}")

            return response

        except Exception as e:
            raise FraudDetectionException(e, sys) from e

    def get_model_pusher_config(self) -> ModelPusherConfig:
        """Get the ModelPusherConfig.
        Returns:
            model_pusher_config(obj:'ModelPusherConfig'): Configuration for model pushing.
        Raises:
            FraudDetectionException: If an error occurs while retrieving the configuration.
        """
        try:
            model_pusher_config_info: ConfigBox = self.config[MODEL_PUSHER_CONFIG_KEY]

            export_dir_path = (
                self.artifact_dir
                / self.config[MODEL_PUSHER_CONFIG_KEY][MODEL_PUSHER_EXPORT_DIR_KEY]
                / self.current_time_stamp
            )
            saved_models_directory = (
                self.artifact_dir
                / model_pusher_config_info[MODEL_PUSHER_EXPORT_DIR_KEY]
            )

            model_pusher_config = ModelPusherConfig(
                model_export_dir=export_dir_path,
                saved_models_directory=saved_models_directory,
            )

            logging.info(f"Logging model pusher: {model_pusher_config}")

        except Exception as e:
            raise FraudDetectionException(e, sys) from e

        return model_pusher_config
