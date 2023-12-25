import os
import sys
from pathlib import Path
from fraudDetection.logger import logging
from fraudDetection.exception import FraudDetectionException
from fraudDetection.entity import DataIngestionArtifact, DataTransformationArtifact, ModelEvaluationConfig, \
    ModelTrainerArtifact, DataValidationArtifact, ModelEvaluationArtifact, evaluate_classification_model
from fraudDetection.utils import read_yaml, write_yaml, load_object, load_data, write_pickle, save_json
from fraudDetection.constants import BEST_MODEL_KEY, HISTORY_KEY, MODEL_PATH_KEY, DATA_SCHEMA_TARGET_COLUMN_KEY, \
    DATA_SCHEMA_COLUMNS_KEY
from fraudDetection.components import processing_data
from sklearn import set_config
import mlflow
from IPython.display import display
from mlflow.models.signature import infer_signature
import pickle
import shutil
from dotenv import load_dotenv

set_config(display='diagram')


class ModelEvaluation:
    def __init__(self, model_evaluation_config: ModelEvaluationConfig, data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_artifact: DataValidationArtifact, model_trainer_artifact: ModelTrainerArtifact,
                 data_transformation_artifact: DataTransformationArtifact) -> None:
        try:
            logging.info(f"{'=' * 20} Model Evaluation Log Started {'=' * 20}")
            self.model_evaluation_config = model_evaluation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.model_trainer_artifact = model_trainer_artifact
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise FraudDetectionException(e, sys) from e

    def get_best_model(self):
        try:
            model_evaluation_dict = {
                BEST_MODEL_KEY: {
                    MODEL_PATH_KEY: str(self.model_trainer_artifact.trained_model_file_path)
                },
            }

            model_evaluation_file_path = Path(self.model_evaluation_config.model_evaluation_file_path)

            if not os.path.exists(model_evaluation_file_path) or os.stat(model_evaluation_file_path).st_size == 0:
                write_yaml(data=model_evaluation_dict, file_path=model_evaluation_file_path )
                return
            else:
                model_eval_file_content = read_yaml(file_path=model_evaluation_file_path)

            logging.info(f'yaml file:{model_evaluation_file_path} loaded successfully')

            model_eval_file_content = dict() if model_eval_file_content is None else model_eval_file_content

            if BEST_MODEL_KEY not in model_eval_file_content:
                return 

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
                previous_best_model = model_eval_content[BEST_MODEL_KEY]

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
            raise FraudDetectionException(e, sys) from e

    def initiate_model_evaluation(self):
        try:
            is_model_accepted = False
          
            run =mlflow.active_run()
            if run:
                logging.info(f"Mlfow run is active {run.info}")
                mlflow.end_run()    

            load_dotenv()
            tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

            if tracking_uri is not None:
                mlflow.set_tracking_uri(tracking_uri)
            else:
                logging.info("Tracking_URI not set")
                return
            
            
            mlflow.start_run(run_name="Evaluation")
                
            # mlflow.set_registry_uri(self.model_evaluation_config.mlflow_uri)
            
            mlflow.set_tags({"version": "v1", "stage": "evaluation"})
            mlflow.log_param("Phase", "eval")

            trained_model_file_path = self.model_trainer_artifact.trained_model_file_path

            is_trained = self.model_trainer_artifact.is_trained
            if not is_trained:
                logging.info("\nTrained model metrics are not as per the base accuracy and threshold specified\n")

            model = self.get_best_model()

            if model is None:
                logging.info("Not found any existing model. Hence accepting trained model")

                model_evaluation_artifact = ModelEvaluationArtifact(
                    is_model_accepted=True,
                    evaluated_model_path=trained_model_file_path
                )

                self.update_evaluation_report(model_evaluation_artifact)

                logging.info(f"Model accepted. Model eval artifact {model_evaluation_artifact}")

                return model_evaluation_artifact
            
            trained_model_object = load_object(trained_model_file_path)
            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path
            os.listdir(train_file_path)
            schema_file_path = self.data_validation_artifact.schema_file_path
            schema = read_yaml(schema_file_path)
            data_schema = schema[DATA_SCHEMA_COLUMNS_KEY]
            target_column_name = schema[DATA_SCHEMA_TARGET_COLUMN_KEY]
            train_df = load_data(train_file_path, data_schema, (50, None))
            test_df = load_data(test_file_path, data_schema, (50, None))
            train_target_feature = train_df[target_column_name]
            test_target_feature = test_df[target_column_name]
            train_df.drop(target_column_name, axis=1, inplace=True)
            test_df.drop(target_column_name, axis=1, inplace=True)
            logging.info(
                f"Original evaluation data shape: train_df {train_df.shape, train_target_feature.shape} test_df {test_df.shape, test_target_feature.shape}")
            
            processor_file_path = self.data_transformation_artifact.processed_object_file_path
            imputed_sampler_file_path = self.data_transformation_artifact.impute_sampler_object_file_path
            processing_obj = load_object(processor_file_path)
            imputed_sampler_obj = load_object(imputed_sampler_file_path)
            logging.info(f"processing pipline {display(processing_obj)}")
            train_arr_with_y, test_arr_with_y = processing_data(
                impute_sampler_obj=imputed_sampler_obj,
                preprocessor_obj=processing_obj,
                train_X=train_df,
                train_y=train_target_feature,
                test_X=test_df,
                test_y=test_target_feature,
            )
            train_x, train_y, test_x, test_y = train_arr_with_y[:, :-1], train_arr_with_y[:, -1], test_arr_with_y[:, :-1], test_arr_with_y[
                                                                                                        :, -1]
            model_list = [model, trained_model_object]

            metric_info_artifact = evaluate_classification_model(
                model_list=model_list,
                X_train=train_x,
                y_train=train_y,
                X_test=test_x,
                y_test=test_y,
                base_score=self.model_trainer_artifact.model_accuracy,
                threshold=self.model_trainer_artifact.threshold
            )
            logging.info(f"Model evaluation completed. model metric artifact: {metric_info_artifact}")            
            
            
            # Serialize the object using pickle
            ser_imputed_sampler_obj = pickle.dumps(imputed_sampler_obj)
            ser_preprocessing_obj = pickle.dumps(processing_obj)

            # Save the serialized object to a file
            temp_root = Path("temp")
            artifact_file_path1 = Path(os.path.join(temp_root,"imputed_sampler_obj.pkl"))
            artifact_file_path2 = Path(os.path.join(temp_root,"ser_preprocessing_obj.pkl"))
            
            write_pickle(file_path=artifact_file_path1,obj=ser_imputed_sampler_obj)
            write_pickle(file_path=artifact_file_path2,obj=ser_preprocessing_obj)
            
            # Log the file containing the serialized object as an artifact
            mlflow.log_artifact(artifact_file_path1)
            mlflow.log_artifact(artifact_file_path2)

            if temp_root.exists():
                shutil.rmtree(temp_root)        
          
            if metric_info_artifact is None:
                response = ModelEvaluationArtifact(is_model_accepted=is_model_accepted,
                                                   evaluated_model_path=trained_model_file_path)
                logging.info(response)
                return response
            
            if metric_info_artifact.model_index == 1:
                is_model_accepted = True
                model_evaluation_artifact = ModelEvaluationArtifact(is_model_accepted=is_model_accepted,
                                                                    evaluated_model_path=trained_model_file_path)
               
                self.update_evaluation_report(model_evaluation_artifact)
                
                logging.info(f"Model accepted. Model eval artifact {model_evaluation_artifact} created")
        
                signature = infer_signature(train_x, train_y)
        
                mlflow.sklearn.log_model(
                    sk_model=metric_info_artifact.model_object,
                    artifact_path="evaluation",
                    registered_model_name=str(metric_info_artifact.model_object.__class__.__name__),
                    signature=signature,
                    input_example=train_x[:1,:],
                    conda_env="conda.yaml",
                    pyfunc_predict_fn="predict",
                    metadata=dict(stage="staging",
                                  is_model_accpeted=is_model_accepted,
                                  model_index=metric_info_artifact.model_index
                                  )
                    )
                
                mlflow.log_params({
                    "base_score": self.model_trainer_artifact.model_accuracy,
                    "threshold": self.model_trainer_artifact.threshold}
                )   
                mlflow.log_metric("train_f1_score", metric_info_artifact.train_f1_score)
                mlflow.log_metric("test_f1_score", metric_info_artifact.test_f1_score)
                mlflow.log_metric("train_fbeta_score", metric_info_artifact.train_fbeta_score)
                mlflow.log_metric("test_fbeta_score", metric_info_artifact.test_fbeta_score)
                mlflow.log_metric("train_roc_auc_score", metric_info_artifact.train_roc_auc_score)
                mlflow.log_metric("test_roc_auc_score", metric_info_artifact.test_roc_auc_score)
                mlflow.log_metric("train_precision_score", metric_info_artifact.train_precision_score)
                mlflow.log_metric("test_precision_score", metric_info_artifact.test_precision_score)
                mlflow.log_metric("train_recall_score", metric_info_artifact.train_recall_score)
                mlflow.log_metric("test_recall_score", metric_info_artifact.test_recall_score)
                mlflow.log_metric("model_accuracy", metric_info_artifact.model_accuracy)
                mlflow.log_metric("train_accuracy_score", metric_info_artifact.train_accuracy_score)
                mlflow.log_metric("test_accuracy_score", metric_info_artifact.test_accuracy_score)
                mlflow.log_metric("model_index", metric_info_artifact.model_index)

                input_schema = {
                    "columns": schema["categorical_columns"] + schema["numerical_columns"]
                }
                mlflow.log_dict(input_schema, 'input_schema.json')
                
                eval_file_path = self.model_evaluation_config.model_evaluation_file_path.parent
                input_schema_file_name = Path("output.json")
                input_schema_file_path = os.path.join(eval_file_path,input_schema_file_name)

                # Save the dictionary to the JSON file
                save_json(input_schema_file_path,input_schema)
      
                if run:
                    logging.info(f"Mlfow run is active {run.info}")
            else:
                logging.info("Trained model is no better than existing model hence not accepting trained model")
                model_evaluation_artifact = ModelEvaluationArtifact(evaluated_model_path=trained_model_file_path,
                                                                    is_model_accepted=False)
            return model_evaluation_artifact

        except Exception as e:
            raise FraudDetectionException(e, sys) from e

    def __del__(self):
        mlflow.end_run()
        logging.info(f"\n{'=' * 20} Model Evaluation Log Completed {'=' * 20}\n\n")
