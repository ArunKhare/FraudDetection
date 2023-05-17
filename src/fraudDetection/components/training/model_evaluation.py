import os, sys
from pathlib import Path
from fraudDetection.logger import logging
from fraudDetection.exception import FraudDetectionException
from fraudDetection.entity import DataIngestionArtifact, DataTransformationArtifact, ModelEvaluationConfig, ModelTrainerArtifact, DataValidationArtifact, ModelEvaluationArtifact, evaluate_classification_model
from fraudDetection.utils import read_yaml, write_yaml, load_object, load_data
from fraudDetection.constants import BEST_MODEL_KEY, HISTORY_KEY, MODEL_PATH_KEY, DATA_SCHEMA_TARGET_COLUMN_KEY, DATA_SCHEMA_COLUMNS_KEY
from fraudDetection.components import processing_data
from box import ConfigBox
import numpy as np
import pandas as pd
from sklearn import set_config
set_config(display='diagram')
from IPython.display import display
import yaml

class ModelEvaluation:
    def __init__(self, model_evaluation_config: ModelEvaluationConfig, data_ingestion_artifact: DataIngestionArtifact, data_validation_artifact: DataValidationArtifact, model_trainer_artifact: ModelTrainerArtifact,data_transformation_artifact: DataTransformationArtifact) -> None:
        try:
            logging.info(f"{'='*20} Model Evaluation Log Started {'='*20}")
            self.model_evaluation_config = model_evaluation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.model_trainer_artifact = model_trainer_artifact
            self.data_validation_artifact = data_validation_artifact 
            self.data_transformation_artifact = data_transformation_artifact
            # self.model_evaluation_artifact = model_evaluation_artifact
        except Exception as e:
            raise FraudDetectionException(e,sys) from e
        
    def get_best_model(self):
        try:
            model_evaluation_dict = {
                BEST_MODEL_KEY: {
                    MODEL_PATH_KEY: str(self.model_trainer_artifact.trained_model_file_path)
                },
            }
          

            model = None
            model_evaluation_file_path = Path(self.model_evaluation_config.model_evaluation_file_path)
        
            if not os.path.exists(model_evaluation_file_path):
                model=write_yaml(file_path=model_evaluation_file_path, data=model_evaluation_dict)
                return model

            model_eval_file_content = None           
            model_eval_file_content = read_yaml(file_path=model_evaluation_file_path)

            logging.info(f'yaml file:{model_evaluation_file_path} loaded successfully' )

            model_eval_file_content = dict() if model_eval_file_content is None else model_eval_file_content
            if BEST_MODEL_KEY not in model_eval_file_content:
                return model

            model = load_object(file_path=Path(model_eval_file_content[BEST_MODEL_KEY][MODEL_PATH_KEY]))

            return model
        except Exception as e:
            raise FraudDetectionException(e, sys) from e
    
    def update_evaluation_report(self, model_evaluation_artifact):
        try:
            eval_file_path = self.model_evaluation_config.model_evaluation_file_path
            model_eval_content = read_yaml(file_path=eval_file_path)
            model_eval_content = dict() if model_eval_content is None else model_eval_content

            previous_best_model = None
            if BEST_MODEL_KEY in model_eval_content:
                preivous_best_model = model_eval_content[BEST_MODEL_KEY]
            
            logging.info(f'Previous eval result: {model_eval_content}')

            eval_result = {
                BEST_MODEL_KEY: {
                    MODEL_PATH_KEY: model_evaluation_artifact.evaluated_model_path
                }
            }

            if previous_best_model is not None:
                model_history = {self.model_evaluation_config.time_stamp: previous_best_model}
                if HISTORY_KEY not in model_eval_content:
                    history = {HISTORY_KEY: model_history}
                    eval_result.update(history)
                else:
                    model_eval_content[HISTORY_KEY].update(model_history)

            model_eval_content.update(eval_result)
            logging.info(f"Updated eval result:{model_eval_content}")
            write_yaml(file_path=eval_file_path, data=model_eval_content)
        except Exception as e:
            raise FraudDetectionException(e,sys) from e
        
    def initiate_model_evaluation(self):
        try:
            trained_model_file_path = self.model_trainer_artifact.trained_model_file_path
            trained_model_object = load_object(trained_model_file_path)

            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path
            os.listdir(train_file_path)
            schema_file_path = self.data_validation_artifact.schema_file_path
            schema = read_yaml(schema_file_path)
            data_schema= schema[DATA_SCHEMA_COLUMNS_KEY]
            target_column_name = schema[DATA_SCHEMA_TARGET_COLUMN_KEY]

            train_df = load_data(train_file_path, data_schema,(-8,-1))
            test_df = load_data(test_file_path, data_schema,(-10,-8))

            train_target_feature = train_df[target_column_name]
            test_target_feature = test_df[target_column_name]

            train_df.drop(target_column_name, axis=1, inplace=True)
            test_df.drop(target_column_name, axis=1, inplace=True)

            logging.info(f"Original Data evaluation shape : train_df {train_df.shape},{train_target_feature.shape} test_df {test_df.shape} {test_target_feature.shape}")

            processor_file_path = self.data_transformation_artifact.processed_object_file_path
            imputer_Sampler_file_path = self.data_transformation_artifact.imputer_sampler_object_file_path

            processing_obj = load_object(processor_file_path)
            imputer_Sampler_obj = load_object(imputer_Sampler_file_path)

            logging.info(f"processing pipline {display(processing_obj)}")

            train_arr_with_y, test_arr_with_y = processing_data(
                imputer_sampler_obj=imputer_Sampler_obj,
                preprocessor_obj=processing_obj,
                train_X=train_df,
                train_y=train_target_feature,
                test_X=test_df,
                test_y=test_target_feature,
                )
            
            train_X , train_y, test_X, test_y = train_arr_with_y[:,:-1], train_arr_with_y[:,-1], test_arr_with_y[:,:-1],test_arr_with_y[:,-1]

            model =self.get_best_model()

            if model is None:
                logging.info ("Not found any existing model. Hence accepting trained model")
                model_evaluation_artifact = ModelEvaluationArtifact(
                    is_model_accepted=True,
                    evaluated_model_path= trained_model_file_path
                )
                self.update_evaluation_report(model_evaluation_artifact)
                logging.info(f"Model accepted. Model eval artifact {model_evaluation_artifact}")
                return model_evaluation_artifact
            model_list = [model, trained_model_object]

            metric_info_artifact = evaluate_classification_model(
                model_list=model_list,
                X_train=train_X,
                y_train=train_y,
                X_test=test_X,
                y_test=test_y,
                base_score=self.model_trainer_artifact.model_accuracy,
                threshold=self.model_trainer_artifact.threshold
            )

            logging.info(f"Model evaluation completed. model metric artifact: {metric_info_artifact}")

            if metric_info_artifact is None:
                response = ModelEvaluationArtifact(is_model_accepted=False,
                                                   evaluated_model_path=trained_model_file_path)
                logging.info(response)
                return response
            if metric_info_artifact.model_index == 1:
                model_evaluation_artifact = ModelEvaluationArtifact(is_model_accepted= True,
                                                                    evaluated_model_path=trained_model_file_path)
                self.update_evaluation_report(model_evaluation_artifact)
                logging.info(f"Model accepted. Model eval artifact {model_evaluation_artifact} created")

            else:
                logging.info("Trained model is no better than existing model hence not accepting trained model")
                model_evaluation_artifact = ModelEvaluationArtifact(evaluated_model_path=trained_model_file_path,
                                                                    is_model_accepted=False)
            return model_evaluation_artifact

        except Exception as e:
            raise FraudDetectionException(e,sys) from e
    
    def __del__(self):
        logging.info(f"\n{'='*20} Model Evaluation Log Completed {'='*20}\n\n")
        