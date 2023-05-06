import os, sys
from typing import List
from pathlib import Path
from fraudDetection.utils import create_directories, load_numpy_array_data, load_data, load_object
from fraudDetection.exception import FraudDetectionException
from fraudDetection.logger import logging
from fraudDetection.entity import ModelTrainerConfig, ModelTrainerArtifact, DataTransformationArtifact
from fraudDetection.utils import load_numpy_array_data, load_object, save_object
from fraudDetection.entity import MetricInfoArtifact, GridSearchedBestModel, ModelFactory
from fraudDetection.entity import evaluate_classification_model


class FraudetectionEstimaorModel:
    def __init__(self, preprocessing_object, trained_model_object) -> None:
        self.preprocessing_object = preprocessing_object
        self.trained_model_object = trained_model_object

    def predict(self,X):
        transformed_feature = self.preprocessing_object.transform(X)
        return self.trained_model_object.predict(transformed_feature)
    def __repr__(self) -> str:
        return f"{type(self.trained_model_object).__name__}()"
    def __str__(self) -> str:
        return f"{type(self.trained_model_object).__name__}()"
    
class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig, data_transformation_artifact:DataTransformationArtifact) -> None:
        try:
            logging.info(f"{'>>'*30} Model training log started {'<<'*30}")
            self.model_trainer_config: ModelTrainerConfig = model_trainer_config
            self.data_transformation_artifact: DataTransformationArtifact = data_transformation_artifact
        except Exception as e:
            raise FraudDetectionException(e,sys) from e
    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            logging.info(f"loading transformed training dataset")
            transformed_train_file_path = Path(self.data_transformation_artifact.transformed_train_file_path)
            train_array = load_numpy_array_data(transformed_train_file_path)

            logging.info(f"Loading transformed testing dataset")
            transformed_test_file_path = Path(self.data_transformation_artifact.transformed_test_file_path)
            test_array = load_numpy_array_data(transformed_test_file_path)

            logging.info(f"Splitting training and testing input and target feature")
            X_train, y_train, X_test, y_test = train_array[:,:-1], train_array[:,-1], test_array[:,:-1], test_array[:,-1]
            
            logging.info(f'Shapes: X_train {X_train}, y_train {y_train}, X_test {X_test}, y_test{y_test}')

            logging.info (f"Extracting model config file path")
            model_config_file_path = self.model_trainer_config.model_config_file_path

            logging.info(f"Initializing model factory class using above model config  file :{model_config_file_path}")

            model_factory = ModelFactory(model_config_path=model_config_file_path)
            base_accuracy = self.model_trainer_config.base_accuracy
            threshold = self.model_trainer_config.threshold_diff_train_test_acc

            logging.info(f"Expected  accuracy: {base_accuracy}, threshold {threshold}")
            
            logging.info(f"Initiating operation model selection")
            best_model = model_factory.get_best_model(X=X_train,y=y_train,base_accuracy=base_accuracy)

            logging.info(f'Best Model found on training dataset: {best_model}')

            logging.info(f"Extracting trained model list")
            grid_searched_best_model_list = model_factory.grid_search_best_model_list

            model_list = [model.best_model for model in grid_searched_best_model_list]
            
            logging.info(f"Evaluation all trained model on training and testing dataset both")

            metric_info:MetricInfoArtifact = evaluate_classification_model(model_list=model_list,X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,base_accuracy=base_accuracy,threshold=threshold)
            print(metric_info)
            logging.info(f"Best model found on both training and testing dataset")

            preprocessing_obj = load_object(file_path=Path(self.data_transformation_artifact.processed_object_file_path))
            model_object = metric_info.model_object
            frauddetection_model = FraudetectionEstimaorModel(preprocessing_object=preprocessing_obj,trained_model_object=model_object)

            trained_model_file_path = Path(self.model_trainer_config.trained_model_file_path)
            logging.info(f"saving model at path: {trained_model_file_path}")

            save_object(
                file_path=trained_model_file_path,
                obj=frauddetection_model
            )

            model_trainer_artifact = ModelTrainerArtifact(
                is_trained=True,
                message="Model Trained Successfully",
                trained_model_file_path=trained_model_file_path,
                train_accuracy=metric_info.train_accuracy_score,
                test_accuracy=metric_info.test_accuracy_score,
                train_f1_score=metric_info.train_f1_score,
                test_f1_score=metric_info.test_f1_score,
                train_precision_score=metric_info.train_precision_score,
                train_recall_score=metric_info.train_recall_score,
                model_accuracy=metric_info.model_accuracy,
                train_accuracy_score=metric_info.train_accuracy_score,
                test_accuracy_score=metric_info.test_accuracy_score
            )
            logging.info(f"Model Trainer Artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        except Exception as e:
            raise FraudDetectionException(e,sys) from e
        
    def __del__(self):
        logging.info(f"{'>>'*30} Model trainer log completed {'<<'*30}")
