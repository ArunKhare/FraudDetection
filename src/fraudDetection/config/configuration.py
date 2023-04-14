from fraudDetection.entity import DataIngestionConfig, DataValidationConfig, DataTransformationConfig,ModelEvaluationConfig, ModelPusherConfig, ModelTrainerConfig, TraningPipelineConfig

from fraudDetection.utils import read_yaml, create_directories
from fraudDetection.logger import logging
import os, sys
from fraudDetection.constants import *
from fraudDetection.exception import FraudDetectionException

class ConfigurationManager:
    def __init__(self, config = CONFIG_FILE_PATH) -> None:
        try:
            self.config_info = read_yaml(config)
            self.current_time_stamp:str = CURRENT_TIME_STAMP
            self.training_pipeline_config = self.get_training_pipeline_config()
            self.artifact_dir = self.training_pipeline_config.artifacts_root
        except Exception as e:
            raise FraudDetectionException(e,sys) from e
        
    def get_training_pipeline_config(self) -> TraningPipelineConfig:
        try:
            training_pipeline_config = self.config_info[TRAINING_PIPELINE_CONFIG_KEY]
            artifact_dir = Path(os.path.join(ROOT_DIR,
                training_pipeline_config[TRAINING_PIPELINE_NAME_KEY],
                training_pipeline_config[TRAINING_PIPELINE_ARTIFACT_DIR_KEY]
                ))
            training_pipline_config = TraningPipelineConfig(artifacts_root=artifact_dir)
        except Exception as e:
            raise FraudDetectionException(e,sys) from e
        return training_pipeline_config
    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        try:
            # artifact_dir = self.training_pipeline_config.artifacts_root
            data_ingestion_artifact_dir = Path(os.path.join(self.artifact_dir,DATA_INGESTION_ARTIFACT_DIR,self.current_time_stamp))
           
            data_ingestion_info = self.config_info[DATA_INGESTION_CONFIG_KEY]
            dataset_download_url = data_ingestion_info[DATA_INGESTION_URL_KEY]
            unzip_dir = Path(os.path.join(data_ingestion_artifact_dir, data_ingestion_info[DATA_INGESTION_UNZIP_DIR_KEY]))
            raw_data_dir = Path(os.path.join(data_ingestion_artifact_dir, data_ingestion_info[DATA_INGESTION_RAW_DATA_DIR_KEY]))
            ingested_data_dir = Path(os.path.join(data_ingestion_artifact_dir,data_ingestion_info[DATA_INGESTION_INGESTED_DIR_KEY]))
            ingested_train_dir = Path(os.path.join(data_ingestion_artifact_dir,data_ingestion_info[DATA_INGESTION_TRAIN_DIR_KEY]))
            ingested_test_dir = Path(os.path.join(data_ingestion_artifact_dir,data_ingestion_info[DATA_INGESTION_TEST_DIR_KEY]))
            data_ingestion_config=DataIngestionConfig(
                source_url=dataset_download_url,
                raw_data_dir=raw_data_dir,
                unzip_dir=unzip_dir,
                ingested_dir=ingested_data_dir,
                ingested_train_dir=ingested_train_dir,
                ingested_test_dir=ingested_test_dir
                )
            logging.info(f"Data ingestion config : {data_ingestion_config}")
        except Exception as e:
            raise FraudDetectionException(e,sys) from e 
        return data_ingestion_config
    

    def get_data_validation_config(self):
        try:
            # artifact_dir= self.training_pipeline_config.artifacts_root
            data_validation_artifacts_dir = Path(os.path.join(self.artifact_dir, DATA_VALIDATION_ARTIFACT_DIR, self.current_time_stamp))
            data_validation_config_info = self.config_info[DATA_VALIDATION_CONFIG_KEY]
            schema_file_name = data_validation_config_info[DATA_VALIDATION_SCHEMA_FILE_NAME_KEY]
            report_file_name = data_validation_config_info[DATA_VALIDATION_REPORT_FILE_NAME_KEY]

            schema_file_path = Path(os.path.join(CONFIG_FILE_PATH))
            data_validation_config = DataValidationConfig(
                schema_dir=schema_file_path,
                schema_file_name = schema_file_name,
                report_file_name = report_file_name
                )
        except Exception as e:
            raise FraudDetectionException(e,sys) from e
        return data_validation_config
    
    def get_data_transformation_config(self):
        try:
            # artifact_dir = self.training_pipeline_config.artifacts_root
            data_transformation_artifacts_dir = Path(os.path.join(self.artifact_dir,DATA_TRANSFORMATION_ARTIFACTS_DIR_KEY,self.current_time_stamp))

            data_transformation_config_info = self.config_info[DATA_TRANSFORMATION_CONFIG_KEY]
            
            transformed_train_dir = Path(os.path.join(data_transformation_artifacts_dir,data_transformation_config_info[DATA_TRANSFORMED_TRAIN_DIR_KEY]))
            transformed_test_dir = Path(os.path.join(data_transformation_artifacts_dir,data_transformation_config_info[DATA_TRANSFORMED_TEST_DIR_KEY]))
            preprocessed_object_file_name = Path(os.path.join(data_transformation_artifacts_dir, data_transformation_config_info[DATA_TRANSFORMATION_PREPROCESSING_DIR_KEY]))

            data_transformation_config = DataTransformationConfig(
                                                tranformed_dir = data_transformation_artifacts_dir,
                                                tranformed_train_dir = transformed_train_dir,
                                                transformed_test_dir = transformed_test_dir,
                                                preprocessed_object_file_name = preprocessed_object_file_name
                                                                  )
        except Exception as e:
            raise FraudDetectionException(e,sys) from e
        logging.info(f"Data transformation config: {data_transformation_config}")
        return data_transformation_config
    
    def get_model_trainer_config(self):
        try:
            model_trainer_artifacts_dir = Path(os.path.join(self.artifact_dir, MODEL_TRAINER_ARTIFACTS_DIR, self.current_time_stamp))

            model_trainer_config_info = self.config_info[MODEL_TRAINER_CONFIG_KEY]
            
            base_accuracy =  model_trainer_config_info[MODEL_TRAINED_BASE_ACCURACY_KEY]
            
            trained_model_file_path = Path(os.path.join(model_trainer_artifacts_dir, model_trainer_config_info[MODEL_TRAINED_DIR_KEY],
            model_trainer_config_info[MODEL_TRAINED_FILE_NAME_KEY]))

            model_config_file_path = Path(os.path.join(model_trainer_config_info[MODEL_TRAINER_MODEL_CONFIG_DIR_KEY],model_trainer_config_info[MODEL_TRAINER_MODEL_CONFIG_FILE_NAME_KEY]))
            
            model_trainer_config = ModelTrainerConfig(
                trained_model_file_path=trained_model_file_path,
                base_accuracy=base_accuracy,
                model_config_file_path=model_config_file_path
                )
        except Exception as e:
            raise FraudDetectionException(e,sys) from e
        logging.info(f"Model trainer config : {model_trainer_config}")
        
        return model_trainer_config

    def get_model_evaluation_config(self):
        try:
            model_evaluation_config = self.config_info[MODEL_EVALUATION_CONFIG_KEY]
            artifact_dir = os.path.join(self.artifact_dir,MODEL_EVALUATION_ARTIFACTS_DIR)

            model_evaluation_file_path = Path(os.path.join(artifact_dir,model_evaluation_config[MODEL_EVALUATION_FILE_NAME_KEY],self.current_time_stamp))

            response = ModelEvaluationConfig(model_evaluation_file_name= model_evaluation_file_path, time_stamp=self.current_time_stamp)

            logging.info(f"Model Evaluation config: {response}")
            return response
        except Exception as e:
            raise FraudDetectionException(e,sys) from e
       
    def get_model_pusher_config(self):
        try:

            model_pusher_config_info = self.config_info[MODEL_PUSHER_CONFIG_KEY]

            export_dir_path = Path(os.path.join(ROOT_DIR,model_pusher_config_info[MODEL_PUSHER_EXPORT_DIR_KEY]),self.current_time_stamp.strftime('%Y%m%d%H%M%S'))

            model_pusher_config = ModelPusherConfig(
                model_export_dir=export_dir_path
            )
        except Exception as e:
            raise FraudDetectionException(e,sys) from e
        