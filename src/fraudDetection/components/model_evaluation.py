import os, sys
from pathlib import Path
from fraudDetection.logger import logging
from fraudDetection.exception import FraudDetectionException
from fraudDetection.entity import DataIngestionArtifact, DataTransformationArtifact, ModelEvaluationConfig, ModelTrainerArtifact, DataValidationArtifact, ModelEvaluationArtifact, evaluate_classification_model
from fraudDetection.utils import read_yaml, write_yaml, load_object, load_data, create_directories
from fraudDetection.constants import BEST_MODEL_KEY, HISTORY_KEY, MODEL_PATH_KEY, DATA_SCHEMA_TARGET_COLUMN_KEY, DATA_SCHEMA_COLUMNS_KEY

from box import ConfigBox
import numpy as np
import pandas as pd
class ModelEvaluation:
    def __init__(self, model_evaluation_config: ModelEvaluationConfig, data_ingestion_artifact: DataIngestionArtifact, data_validation_artifact: DataValidationArtifact, model_trainer_artifact: ModelTrainerArtifact) -> None:
        try:
            logging.info(f"{'>>'*30} Model Evaluation Log Started {'<<'*30}")
            self.model_evaluation_config = model_evaluation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.model_trainer_artifact = model_trainer_artifact
            self.data_validation_artifact = data_validation_artifact 
            # self.model_evaluation_artifact = model_evaluation_artifact
        except Exception as e:
            raise FraudDetectionException(e,sys) from e
        
    def get_best_model(self):
        try:
            model_evaluation = {
                BEST_MODEL_KEY: {
                    MODEL_PATH_KEY: str(self.model_trainer_artifact.trained_model_file_path)
                },
            }

            model = None
            model_evaluation_file_path = Path(self.model_evaluation_config.model_evaluation_file_path)
            
            if not os.path.exists(model_evaluation_file_path):
                write_yaml(file_path=model_evaluation_file_path, data = model_evaluation)
                return model

            model_eval_file_content = None           
            model_eval_file_content = read_yaml(file_path=model_evaluation_file_path)
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

            train_df = load_data(train_file_path, data_schema,-3,-1)
            test_df = load_data(test_file_path, data_schema,-5,-3)

            train_target_arr = np.array(train_df[target_column_name])
            test_target_arr = np.array(test_df[target_column_name])

            train_df.drop(target_column_name, axis=1, inplace=True)
            test_df.drop(target_column_name, axis=1, inplace=True)

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
                X_train=train_df,
                y_train= train_target_arr,
                X_test=test_df,
                y_test=test_target_arr,
                base_accuracy=self.model_trainer_artifact.model_accuracy,
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
        