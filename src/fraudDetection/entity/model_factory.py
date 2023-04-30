from cmath import log
import os, sys
import importlib
from pyexpat import model
import numpy as np
import yaml

from typing import List
from sklearn.metrics import f1_score, precision_recall_curve, recall_score, classification_report, fbeta_score, label_ranking_average_precision_score,  precision_score ,accuracy_score, balanced_accuracy_score
from fraudDetection.exception import FraudDetectionException
from fraudDetection.logger import logging

from fraudDetection.constants import FACTORY_CLASS_KEY, FACTORY_GRID_SEARCH_KEY, FACTORY_MODEL_SELECTION_KEY, FACTORY_MODULE_KEY, FACTORY_SEARCH_PARAM_GRID_KEY, FACTORY_PARAMS

from fraudDetection.entity import InitializedModelDetails, GridSearchBestModel, BestModel, MetricInfoArtifact


def evaluate_classification_model(model_list: list, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, base_accuracy: float=0.6) -> MetricInfoArtifact:

    try:
        index_number = 0
        metric_info_artifact = None
        for model in model_list:
            model_name = str(model)
            logging.info(f"{'>>'*30} Starting evaluating model:{type(model).__name__}{'<<'*30} ")

            #getting prediction for training and testing dataset
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            #calculating f1 score on training and testing dataset
            train_f1_score = f1_score(y_train,y_train_pred)
            test_f1_score = f1_score(y_test,y_test_pred)
            #calculating precision  score for training and testing dataset
            train_precision_score = precision_score(y_train,y_train_pred)
            test_precision_score = precision_score(y_test, y_test_pred)
            #calculating recall score on training and testing dataset
            train_recall_score = recall_score(y_train,y_train_pred)
            test_recall_score = recall_score(y_test,y_test_pred)
            #calculating label ranking avg precision score on training and testing dataset
            train_label_ranking_avg_prec_score = label_ranking_average_precision_score(y_train,y_train_pred)
            test_label_ranking_aveg_prec_score = label_ranking_average_precision_score(y_test,y_test_pred)
            #getting classification report on training and testing dataset
            train_classification_report = classification_report(y_train,y_train_pred)
            test_classification_report = classification_report(y_test,y_test_pred)

            train_precision_recall_curve = precision_recall_curve(y_train, y_train_pred)
            test_precision_recall_curve = precision_recall_curve(y_test, y_test_pred)
            
            # Accuracy Score on train and test dataset 
            train_accuracy_score = accuracy_score(y_train,y_train_pred)
            test_accuracy_score = accuracy_score(y_test,y_test_pred)

            #logging all important metric
            logging.info(f"{'>>'*30}Score {'<<'*30}")
            logging.info(f"F1 Score: train {train_f1_score}, test {test_f1_score}")
            logging.info(f"Precision Score :  train {train_precision_score} test {test_precision_score}")
            logging.info(f"Recall Score : train {train_recall_score} test {test_recall_score}" )
            logging.info(f"Label Ranking Avg Precision Score: train {train_label_ranking_avg_prec_score} \n test {test_label_ranking_aveg_prec_score}")
            logging.info(f"Classification Report : train {train_classification_report} test {test_classification_report}")
            logging.info(f"Precision Recall Curve: train {train_precision_recall_curve}, test {test_precision_recall_curve}")
            logging.info(f'Accuracy Score: train  {train_accuracy_score} test {test_recall_score}')
            
            # calculating harmonic mean of train and test accuracy_score
            model_accuracy = (2*train_accuracy_score*test_accuracy_score)/(train_accuracy_score + test_accuracy_score)
            diff_test_train_acc = abs(train_accuracy_score-test_accuracy_score)

            logging.info( f'Diff. test train accuracy: {diff_test_train_acc}')
            logging.info( f'Model accuracy: {model_accuracy}')
            # defining threshold

            if model_accuracy >=base_accuracy and diff_test_train_acc <0.05:
                base_accuracy = model_accuracy
                metric_info_artifact = MetricInfoArtifact(
                    model_name=model_name,
                    model_object=model,
                    train_f1_score=train_f1_score,
                    test_f1_score=test_f1_score,
                    train_precision_score=train_precision_score,
                    train_recall_score=train_recall_score,
                    train_label_rank_avg_precision_curve= train_label_ranking_avg_prec_score,
                    model_accuracy=model_accuracy,
                    model_index= index_number
                )
                logging.info(f'Acceptable model found {metric_info_artifact} ')
            index_number +=1
        if metric_info_artifact is None:
            logging.info(f"No model with higher accuracy than base accuracy")
    except Exception as e:
        raise FraudDetectionException(e,sys) from e

def get_sample_model_config_yaml_file(export_dir: str):
    try:
        model_config = {
            FACTORY_GRID_SEARCH_KEY:{
                FACTORY_MODULE_KEY: "sklearn.model_selection",
                FACTORY_CLASS_KEY:"GridSearchCV",
                FACTORY_PARAMS:{
                    "cv": 3,
                    "verbose" : 1
                    }
            },
            FACTORY_MODEL_SELECTION_KEY:{
                "module_0": {
                    FACTORY_MODULE_KEY:"module_of_model",
                    FACTORY_CLASS_KEY: "ModelClassNAme",
                    FACTORY_PARAMS:{
                        "param_name1": "value1",
                        "param_name2": "value2",
                    },
                    FACTORY_SEARCH_PARAM_GRID_KEY:{
                        "param_name":['param_value1','param_value2']
                    }
                },
            }
        }
        os.makedirs(export_dir,exist_ok=True)
        export_file_path = os.path.join(export_dir, "model.yaml")
        with open(export_file_path, 'w') as files:
            yaml.dump(model_config, files)
        return export_file_path
    except Exception as e:
        raise FraudDetectionException(e,sys) from e
    

class ModelFactory:
    def __init__(self, model_config_path: str=None,) -> None:
        try:
            self.config:dict = ModelFactory.read_params(model_config_path)
            self.grid_serach_cv_module = self.config[FACTORY_GRID_SEARCH_KEY][FACTORY_MODULE_KEY]
            self.grid_class_name = self.config[FACTORY_GRID_SEARCH_KEY[FACTORY_CLASS_KEY]]
            self.grid_search_property_data = dict(self.config[FACTORY_GRID_SEARCH_KEY][FACTORY_PARAMS])

            self.initialized_model_list = None
            self.grid_search_best_model_list = None
        except Exception as e:      
            raise FraudDetectionException(e,sys) from e
        
    @staticmethod
    def update_property_of_class(instance_ref: object, property_data: dict):
        try:
            
        except Exception as e:
            raise FraudDetectionException(e,sys) from e
        
    @staticmethod
    def read_params(config_path: str) -> dict:
        try:
            with open(config_path) as yaml_file:
                config:dict = yaml.safe_load(yaml_file)
            return config
        except Exception as e:
            raise FraudDetectionException(e,sys)
        
    @staticmethod
    def class_for_name(module_name: str, class_name: str):
        try:
        except Exception as e:
            raise FraudDetectionException(e,sys) from e
        
  
    def execute_grid_search_operation(self, initialized_model: InitializedModelDetails, output_feature) -> GridSearchBestModel:
        
    try:

    except Exception as e:
        raise FraudDetectionException(e,sys) from e
        
    def get_initialized_model_list(self):

    try:
    except Exception as e
        raise FraudDetectionException from e
    
    def initiate_best_parameter_search_for_initialized_model(self, initialized_model_list, input_feature, output_feature):
        try:
        except Exception as e:
            raise FraudDetectionException(e,sys) from e
        
      def initiate_best_parameter_search_for_initialized_models(self,
                                                              initialized_model_list: List[InitializedModelDetail],
                                                              input_feature,
                                                              output_feature) -> List[GridSearchedBestModel]:
try:
except Exception as e:
    raise FraudDetectionException(e,sys) from e

@staticmethod
def get_model_detail(self):
    pass
def get_best_model_from_grid_searched_best_model_list(self):
    pass
def get_best_model(self):
    pass
