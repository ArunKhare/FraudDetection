from pathlib import Path
from typing import Dict
import os, sys
from box import ConfigBox

from fraudDetection.entity import (
    DataIngestionConfig, 
    DataValidationConfig, 
    DataTransformationConfig,
    ModelEvaluationConfig, 
    ModelPusherConfig, 
    ModelTrainerConfig, 
    TraningPipelineConfig
)

from fraudDetection.utils import read_yaml
from fraudDetection.logger import logging
from fraudDetection.constants import *
from fraudDetection.exception import FraudDetectionException

class ConfigurationManager:
    def __init__(self, config:Path=CONFIG_FILE_PATH) -> None:
        try:
            self.config:ConfigBox = read_yaml(config)
            self.current_time_stamp:str = CURRENT_TIME_STAMP
            self.training_pipeline_config: TraningPipelineConfig = self.get_training_pipeline_config()
            self.artifact_dir: Path = self.training_pipeline_config.artifacts_root
        except Exception as e:
            raise FraudDetectionException(e,sys) from e
        
    def get_training_pipeline_config(self) -> TraningPipelineConfig:
        try:
            training_pipeline:ConfigBox = self.config[TRAINING_PIPELINE_CONFIG_KEY]
            artifact_dir = Path(os.path.join(ROOT_DIR,
                # training_pipeline[TRAINING_PIPELINE_NAME_KEY],
                training_pipeline[TRAINING_PIPELINE_ARTIFACT_DIR_KEY]
            ))
            training_pipeline_config = TraningPipelineConfig(artifacts_root=artifact_dir)
        except Exception as e:
            raise FraudDetectionException(e,sys) from e
        return training_pipeline_config
    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        try:
            data_ingestion_artifact_dir = Path(os.path.join(self.artifact_dir,DATA_INGESTION_ARTIFACT_DIR_KEY))
                                            #    ,self.current_time_stamp))
           
            data_ingestion_info:ConfigBox = self.config[DATA_INGESTION_CONFIG_KEY]
            dataset_download_url:ConfigBox = data_ingestion_info[DATA_INGESTION_URL_KEY]
            zip_dir = Path(os.path.join(data_ingestion_artifact_dir, data_ingestion_info[DATA_INGESTION_ZIP_DIR_KEY]))
            raw_data_dir = Path(os.path.join(data_ingestion_artifact_dir, data_ingestion_info[DATA_INGESTION_RAW_DATA_DIR_KEY]))
            ingested_data_dir = Path(os.path.join(data_ingestion_artifact_dir,data_ingestion_info[DATA_INGESTION_INGESTED_DIR_KEY]))
            ingested_train_dir = Path(os.path.join(data_ingestion_artifact_dir,data_ingestion_info[DATA_INGESTION_TRAIN_DIR_KEY]))
            ingested_test_dir = Path(os.path.join(data_ingestion_artifact_dir,data_ingestion_info[DATA_INGESTION_TEST_DIR_KEY]))
            stratify:str = data_ingestion_info[DATA_INGESTION_STRATIFY_COL_KEY]
            test_size:float = data_ingestion_info[DATA_INGESTION_TEST_SIZE_KEY]

            data_ingestion_config = DataIngestionConfig(
                source_url=dataset_download_url,
                raw_data_dir=raw_data_dir,
                zip_dir=zip_dir,
                ingested_dir=ingested_data_dir,
                ingested_train_dir=ingested_train_dir,
                ingested_test_dir=ingested_test_dir,
                stratify=stratify,
                test_size=test_size,
            )
            logging.info(f"Data ingestion config : {data_ingestion_config}")
        except Exception as e:
            raise FraudDetectionException(e,sys) from e 
        
        return data_ingestion_config

    def get_data_validation_config(self) -> DataValidationConfig:
        try:
            # artifact_dir= self.training_pipeline_config.artifacts_root
            data_validation_artifacts_dir = Path(os.path.join(self.artifact_dir, DATA_VALIDATION_ARTIFACT_DIR_KEY, self.current_time_stamp))
            data_validation_config:ConfigBox = self.config[DATA_VALIDATION_CONFIG_KEY]
            schema_file_name:str = data_validation_config[DATA_VALIDATION_SCHEMA_FILE_NAME_KEY]
            schema_file_path = Path(os.path.join(ROOT_DIR, CONFIG_DIR, schema_file_name))
            report_file_name:str = data_validation_config[DATA_VALIDATION_REPORT_FILE_NAME_KEY]
            report_file_path = Path(os.path.join(data_validation_artifacts_dir,report_file_name))
            
            data_validation_config =  DataValidationConfig(
                schema_file_name = schema_file_name,
                schema_file_path = schema_file_path,
                report_file_name = report_file_name,
                report_file_path = report_file_path,
                )
            
            logging.info(f"logging data validation config: {data_validation_config}")
        except Exception as e:
            raise FraudDetectionException(e,sys) from e
        return data_validation_config
    
    def get_data_transformation_config(self) -> DataTransformationConfig:
        try:
            # artifact_dir = self.training_pipeline_config.artifacts_root
            data_transformation_artifacts_dir = Path(os.path.join(self.artifact_dir,DATA_TRANSFORMATION_ARTIFACTS_DIR_KEY))

            data_transformation_config:ConfigBox = self.config[DATA_TRANSFORMATION_CONFIG_KEY]
            
            transformed_train_dir = Path(os.path.join(data_transformation_artifacts_dir,data_transformation_config[DATA_TRANSFORMED_TRAIN_DIR_KEY]))
            transformed_test_dir = Path(os.path.join(data_transformation_artifacts_dir,data_transformation_config[DATA_TRANSFORMED_TEST_DIR_KEY]))
            preprocessed_object_dir = Path(os.path.join(data_transformation_artifacts_dir, data_transformation_config[DATA_TRANSFORMATION_PREPROCESSING_DIR_KEY]))


            data_transformation_config = DataTransformationConfig(
                                                transformed_dir = data_transformation_artifacts_dir,
                                                transformed_train_dir = transformed_train_dir,
                                                transformed_test_dir = transformed_test_dir,
                                                preprocessing_object_dir =  preprocessed_object_dir
                                                                  )
        except Exception as e:
            raise FraudDetectionException(e,sys) from e
        logging.info(f"Data transformation config: {data_transformation_config}")
        return data_transformation_config
    
    def get_model_trainer_config(self) -> ModelTrainerConfig:
        try:
            model_trainer_artifacts_dir = Path(os.path.join(self.artifact_dir, MODEL_TRAINER_ARTIFACTS_DIR_KEY, self.current_time_stamp))

            model_trainer_config:ConfigBox = self.config[MODEL_TRAINER_CONFIG_KEY]
            
            base_accuracy:ConfigBox =  model_trainer_config[MODEL_TRAINED_BASE_ACCURACY_KEY]
            
            trained_model_file_path = Path(os.path.join(model_trainer_artifacts_dir, model_trainer_config[MODEL_TRAINED_DIR_KEY],
            model_trainer_config[MODEL_TRAINED_FILE_NAME_KEY]))

            model_config_file_path = Path(os.path.join(model_trainer_config[MODEL_TRAINER_MODEL_CONFIG_DIR_KEY],model_trainer_config[MODEL_TRAINER_MODEL_CONFIG_FILE_NAME_KEY]))
            
            model_trainer_config = ModelTrainerConfig(
                trained_model_file_path=trained_model_file_path,
                base_accuracy=base_accuracy,
                model_config_file_path=model_config_file_path
                )
        except Exception as e:
            raise FraudDetectionException(e,sys) from e
        logging.info(f"Model trainer config : {model_trainer_config}")
        
        return model_trainer_config

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        try:
            model_evaluation_config:ConfigBox = self.config[MODEL_EVALUATION_CONFIG_KEY]
            artifact_dir:ConfigBox = os.path.join(self.artifact_dir,MODEL_EVALUATION_ARTIFACTS_DIR_KEY)

            model_evaluation_file_path = Path(os.path.join(artifact_dir,model_evaluation_config[MODEL_EVALUATION_FILE_NAME_KEY],self.current_time_stamp))

            response = ModelEvaluationConfig(model_evaluation_file_name= model_evaluation_file_path, time_stamp=self.current_time_stamp)
            logging.info(f"Model Evaluation config: {response}")
            return response
        except Exception as e:
            raise FraudDetectionException(e,sys) from e
       
    def get_model_pusher_config(self) -> None:
        try:
            model_pusher_config:ConfigBox = self.config[MODEL_PUSHER_CONFIG_KEY]

            export_dir_path = Path(os.path.join(ROOT_DIR,model_pusher_config[MODEL_PUSHER_EXPORT_DIR_KEY]),self.current_time_stamp.strftime('%Y%m%d%H%M%S'))

            model_pusher_config = ModelPusherConfig(
                model_export_dir=export_dir_path
            )
            logging.info(f"Logging model pusher: {model_pusher_config}")
        except Exception as e:
            raise FraudDetectionException(e,sys) from e
        