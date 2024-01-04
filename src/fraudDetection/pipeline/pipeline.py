"""
This module defines a threaded training pipeline for the Fraud Detection System. It integrates various components, including data ingestion, validation, transformation, model training, evaluation, and pushing. The pipeline is configured using the ConfigurationManager class, and the experiment details are recorded and saved.

Module Components:
    - Pipeline: Class representing the threaded experiment pipeline.
    - ConfigurationManager: Class for managing configuration settings.
    - DataIngestion, DataValidation, DataTransformation, ModelTrainer, ModelEvaluation, ModelPusher: Components responsible for different stages of the pipeline.
    - Other dependencies: os, sys, uuid, datetime, Thread, pandas, logging.

Attributes:
    - experiment: Class attribute representing an instance of the Experiment class with 11 parameters initialized to None.
    - experiment_file_path: Class attribute representing the path to the experiment file, initially set to None.

Configuration Setup:
    - Initializes the ConfigurationManager with the specified configuration file path.
    - Retrieves configurations for data ingestion, validation, transformation, model training, evaluation, and pushing.

Pipeline Stages:
    1. Data Ingestion: Initializes and executes the data ingestion component.
    2. Data Validation: Initializes and executes the data validation component.
    3. Data Transformation: Initializes and executes the data transformation component.
    4. Model Training: Initializes and executes the model training component.
    5. Model Evaluation: Initializes and executes the model evaluation component.
    6. Model Pushing: Initializes and executes the model pushing component.

Experiment Recording:
    - Creates an instance of the Experiment class to record experiment details.
    - Saves experiment details to a CSV file.

Experiment Retrieval:
    - Provides a method (get_experiment_status) to retrieve experiment status data.

Note: This module is meant to be executed directly (__name__ == "__main__") to run the training pipeline.

"""

import os
import sys
import uuid
from datetime import datetime
from threading import Thread

import pandas as pd

from fraudDetection.components import (
    DataIngestion,
    DataValidation,
    DataTransformation,
    ModelTrainer,
    ModelEvaluation,
    ModelPusher,
)

from fraudDetection.config.configuration import ConfigurationManager, CONFIG_FILE_PATH
from fraudDetection.constants import EXPERIMENT_FILE_NAME, EXPERIMENT_DIR_NAME
from fraudDetection.entity import (
    DataIngestionArtifact,
    ModelEvaluationArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact,
    DataValidationArtifact,
)

from fraudDetection.entity import Experiment
from fraudDetection.exception import FraudDetectionException
from fraudDetection.logger import logging
from fraudDetection.utils import create_directories


class Pipeline(Thread):
    """This class initiates a new pipeline as a threaded experiment.

    Attributes:
        experiment (Experiment): An instance of the Experiment class with 11 parameters initialized to None.
        experiment_file_path (str): Path to the experiment file, initially set to None.

    Args:
        Thread (class): The standard Python class used as the superclass for threading.
    """

    experiment = Experiment(*([None] * 11))
    experiment_file_path = None

    def __init__(self, config: ConfigurationManager(CONFIG_FILE_PATH)):
        """intialize the obj with object CongigurationManager
        Args:
            config (obj:'ConfigurationManager'): contains the path for dirs of artifacts
        """
        try:
            artifact_dir = config.training_pipeline_config.artifacts_root
            create_directories([artifact_dir])
            Pipeline.experiment_file_path = os.path.join(
                artifact_dir, EXPERIMENT_DIR_NAME, EXPERIMENT_FILE_NAME
            )
            super().__init__(name="pipeline", daemon=False)
            self.config = config

        except Exception as e:
            raise FraudDetectionException(e, sys) from e

    def start_data_ingestion(self) -> DataIngestionArtifact:
        """Creates directory's for data ingestion in to the artifacts
        Returns:
            DataIngestionArtifact: An instance of the DataIngestionArtifact class representing the data ingestion process.
        """
        try:
            data_ingestion = DataIngestion(
                data_ingestion_config=self.config.get_data_ingestion_config()
            )

            return data_ingestion.initiate_data_ingestion()

        except Exception as e:
            raise FraudDetectionException(e, sys) from e

    def start_data_validation(
        self, data_ingestion_artifact=DataIngestionArtifact
    ) -> DataValidationArtifact:
        """Creates directory's for data  validation reports artifacts
        Args:
            data_ingestion_artifact (obj:'DataIngestionArtifact'): reference to train test csv file
        Returns:
            DataValidationArtifact: An instance of the DataValidationArtifact class representing the data validation process.
        """
        try:
            data_validation = DataValidation(
                data_validation_config=self.config.get_data_validation_config(),
                data_ingestion_artifact=data_ingestion_artifact,
            )

            return data_validation.initiate_data_validation()

        except Exception as e:
            raise FraudDetectionException(e, sys) from e

    def start_data_transformation(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_artifact: DataValidationArtifact,
    ):
        """Creates directory's for data transformation artifacts
        Args:
            data_ingestion_artifact (obj:'DataIngestionArtifact'): reference to train test csv files
            data_validation_artifact (obj:'DataValidationArtifact'): reference to reports in directory
        Returns:
           DataTransformationArtifact: An instance of DataTransformationArtifact class representing the data transformation process
        """
        try:
            data_transformation = DataTransformation(
                data_transformation_config=self.config.get_data_transformation_config(),
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_artifact=data_validation_artifact,
            )

            return data_transformation.initiate_data_transformation()

        except Exception as e:
            raise FraudDetectionException(e, sys) from e

    def start_model_trainer(
        self, data_transformation_artifact: DataTransformationArtifact
    ):
        """Creates directory's for Model trainer artifacts
        Args:
            data_transformation_artifact (obj:'DataTransformationArtifact'): reference to ransformed data artifact path
        Returns:
            ModelTrainerArtifact: An instance of ModelTrainerArtifact class representing  the model trainer process
        """
        try:
            model_trainer = ModelTrainer(
                model_trainer_config=self.config.get_model_trainer_config(),
                data_transformation_artifact=data_transformation_artifact,
            )
            return model_trainer.initiate_model_trainer()

        except Exception as e:
            raise FraudDetectionException(e, sys) from e

    def start_model_evaluation(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_artifact: DataValidationArtifact,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_artifact: ModelTrainerArtifact,
    ):
        """Creates directory's for Model evaluation artifacts
        Args:
            data_ingestion_artifact (obj:'DataIngestionArtifact'): reference to train test csv files
            data_validation_artifact (obj:'DataValidationArtifact'): reference to reports in directory
            data_transformation_artifact (obj:'DataTransformationArtifact'): reference to ransformed data artifact path
            model_trainer (obj:'ModelTrainer'): Reference to the paths created for  the Trained model artifacts
        Returns:
            ModelEvaluationArtifact: An instance of ModelTrainerArtifact representing the model evaluation process
        """
        try:
            model_evaluation = ModelEvaluation(
                model_evaluation_config=self.config.get_model_evaluation_config(),
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_artifact=data_validation_artifact,
                model_trainer_artifact=model_trainer_artifact,
                data_transformation_artifact=data_transformation_artifact,
            )

            return model_evaluation.initiate_model_evaluation()

        except Exception as e:
            raise FraudDetectionException(e, sys) from e

    def start_model_pusher(self, model_eval_artifact: ModelEvaluationArtifact):
        """Create directory for Model pusher artifacts
        Args:
            model_eval_artifacts (obj:'ModelEvaluationArtifact'): reference to model evalution paths
        Returns:
            ModelPusherArtifacts: An instance of ModelPusherArtifacts representing the model pusher process
        """
        try:
            model_pusher = ModelPusher(
                model_pusher_config=self.config.get_model_pusher_config(),
                model_evaluation_artifact=model_eval_artifact,
            )

            return model_pusher.initiate_model_pusher()

        except Exception as e:
            raise FraudDetectionException(e, sys) from e

    def run_pipeline(self):
        """Run pipeline creating a new instance of experiment a process of training and evaluting models"""
        try:
            if Pipeline.experiment.running_status:
                logging.info("Pipeline is already running")

            # data ingestion
            logging.info("Pipeline starting")

            experiment_id = str(uuid.uuid4())

            Pipeline.experiment = Experiment(
                experiment_id=experiment_id,
                initialization_timestamp=self.config.current_time_stamp,
                artifact_timestamp=self.config.current_time_stamp,
                running_status=True,
                start_time=datetime.now(),
                stop_time=None,
                execution_time=None,
                experiment_file_path=Pipeline.experiment_file_path,
                is_model_accepted=None,
                message="Pipe has been started",
                accuracy=None,
            )

            logging.info(f"Pipeline experiment: {Pipeline.experiment}")

            self.save_experiment()
            data_ingestion_artifact = self.start_data_ingestion()

            data_validation_artifact = self.start_data_validation(
                data_ingestion_artifact=data_ingestion_artifact
            )

            data_transformation_artifact = self.start_data_transformation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_artifact=data_validation_artifact,
            )

            model_trainer_artifact = self.start_model_trainer(
                data_transformation_artifact=data_transformation_artifact
            )

            model_evaluator_artifact = self.start_model_evaluation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_artifact=data_validation_artifact,
                data_transformation_artifact=data_transformation_artifact,
                model_trainer_artifact=model_trainer_artifact,
            )

            # if model_evaluator_artifact.is_model_accepted:
            model_pusher_artifact = self.start_model_pusher(
                model_eval_artifact=model_evaluator_artifact
            )
            logging.info(f"Model pusher artifact: {model_pusher_artifact}")
            # else:
            #     logging.info("Trained model rejected")

            logging.info("Pipline completed")

            stop_time = datetime.now()

            Pipeline.experiment = Experiment(
                experiment_id=Pipeline.experiment.experiment_id,
                initialization_timestamp=self.config.current_time_stamp,
                artifact_timestamp=self.config.current_time_stamp,
                running_status=False,
                start_time=Pipeline.experiment.start_time,
                stop_time=stop_time,
                execution_time=stop_time - Pipeline.experiment.start_time,
                message="Pipeline has been completed",
                experiment_file_path=Pipeline.experiment_file_path,
                is_model_accepted=model_evaluator_artifact.is_model_accepted,
                accuracy=model_trainer_artifact.model_accuracy,
            )

            logging.info(f"Pipeline experiment:{Pipeline.experiment}")
            self.save_experiment()

        except Exception as e:
            raise FraudDetectionException(e, sys) from e

    def run(self):
        """Run the pipeline of training and testing models"""
        try:
            self.run_pipeline()

        except Exception as e:
            raise FraudDetectionException(e, sys) from e

    @staticmethod
    def save_experiment():
        """Save the experiments attributes in a csv file record"""
        try:
            if Pipeline.experiment.experiment_id is not None:
                experiment = Pipeline.experiment
                experiment_dict = experiment.__dict__
                experiment_dict = {
                    key: [value] for key, value in experiment_dict.items()
                }

                experiment_dict.update(
                    {
                        "create_time_stamp": [datetime.now()],
                        "experiment_file_path": [
                            os.path.basename(Pipeline.experiment_file_path)
                        ],
                    }
                )

                experiment_report = pd.DataFrame(experiment_dict)

                create_directories([os.path.dirname(Pipeline.experiment_file_path)])

                if os.path.exists(Pipeline.experiment_file_path):
                    experiment_report.to_csv(
                        Pipeline.experiment_file_path,
                        index=False,
                        header=False,
                        mode="a",
                    )
                else:
                    os.path.exists(Pipeline.experiment_file_path)
                    experiment_report.to_csv(
                        Pipeline.experiment_file_path,
                        index=False,
                        header=True,
                        mode="w",
                    )

            else:
                print("First start experiment")

        except Exception as e:
            raise FraudDetectionException(e, sys) from e

    @classmethod
    def get_experiment_status(cls, limit: int = 5) -> pd.DataFrame:
        """get status of experiments
        Returns:
            DataFrame (obj:'pd.DataFrame'): consist of exepriments of training and testing models conducted so far
        """
        try:
            if os.path.exists(Pipeline.experiment_file_path):
                df = pd.read_csv(Pipeline.experiment_file_path)
                limit = -1 * int(limit)

                return df[limit:].drop(
                    columns=["experiment_file_path", "initialization_timestamp"], axis=1
                )
            else:
                return pd.DataFrame()

        except Exception as e:
            raise FraudDetectionException(e, sys) from e
