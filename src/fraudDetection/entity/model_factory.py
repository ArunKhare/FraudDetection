import sys
import importlib
from pathlib import  Path
import mlflow.sklearn
import numpy as np
import yaml
from mlflow import MlflowClient
from mlflow.entities import ViewType
from typing import List
from sklearn.metrics import f1_score, fbeta_score, roc_auc_score, precision_recall_curve, recall_score, \
    classification_report, precision_score, accuracy_score, make_scorer, auc
from fraudDetection.exception import FraudDetectionException
from fraudDetection.logger import logging

from fraudDetection.constants import FACTORY_CLASS_KEY, FACTORY_GRID_SEARCH_KEY, FACTORY_MODEL_SELECTION_KEY, \
    FACTORY_MODULE_KEY, FACTORY_SEARCH_PARAM_GRID_KEY, FACTORY_PARAMS_KEY

from fraudDetection.entity import InitializedModelDetails, GridSearchedBestModel, BestModel, MetricInfoArtifact


def evaluate_classification_model(model_list: list, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray,
                                  y_test: np.ndarray, base_score: float = 0.6,
                                  threshold: float = 0.05) -> MetricInfoArtifact:
    try:
        index_number = 0
        metric_info_artifact = None
        initial_base_score = 0
        for model in model_list:
            model_name = str(model)
            logging.info(f"\n{'>>' * 20} Starting evaluating model:{type(model).__name__}{'<<' * 20} ")

            # getting prediction for training and testing dataset
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # calculating f1 score on training and testing dataset
            train_f1_score = f1_score(y_train, y_train_pred)
            test_f1_score = f1_score(y_test, y_test_pred)

            # calculating f-beta-score on training and testing dataset
            train_fbeta_score = fbeta_score(y_train, y_train_pred, beta=2)
            test_fbeta_score = fbeta_score(y_test, y_test_pred, beta=2)

            # calculating f-betas-core on training and testing dataset
            train_roc_auc_score = roc_auc_score(y_train, y_train_pred)
            test_roc_auc_score = roc_auc_score(y_test, y_test_pred)

            # calculating precision  score for training and testing dataset
            train_precision_score = precision_score(y_train, y_train_pred)
            test_precision_score = precision_score(y_test, y_test_pred)
            # calculating recall score on training and testing dataset
            train_recall_score = recall_score(y_train, y_train_pred)
            test_recall_score = recall_score(y_test, y_test_pred)

            # getting classification report on training and testing dataset
            train_classification_report = classification_report(y_train, y_train_pred)
            test_classification_report = classification_report(y_test, y_test_pred)

            train_precision_recall_curve = precision_recall_curve(y_train, y_train_pred)
            test_precision_recall_curve = precision_recall_curve(y_test, y_test_pred)

            # Accuracy Score on train and test dataset 
            train_accuracy_score = accuracy_score(y_train, y_train_pred)
            test_accuracy_score = accuracy_score(y_test, y_test_pred)

            # logging all important metric
            logging.info(f"\n{'>>' * 20}Score {'<<' * 20}")
            logging.info(f"F1 Score: train {train_f1_score}, test {test_f1_score}")
            logging.info(f"F-beta Score: train {train_fbeta_score}, test {test_fbeta_score}")
            logging.info(f"Roc_Auc Score: train {train_roc_auc_score}, test {test_roc_auc_score}")
            logging.info(f"Precision Score :  train {train_precision_score} test {test_precision_score}")
            logging.info(f"Recall Score : train {train_recall_score} test {test_recall_score}")
            logging.info(
                f"Classification Report : train \n {train_classification_report} test \n {test_classification_report}")
            logging.info(
                f"Precision Recall Curve: train {train_precision_recall_curve}, test {test_precision_recall_curve}")
            logging.info(f'Accuracy Score: train  {train_accuracy_score} test {test_recall_score}')

            # calculating harmonic mean of train and test accuracy_score
            model_accuracy = (2 * train_accuracy_score * test_accuracy_score) / (
                    train_accuracy_score + test_accuracy_score)
            diff_test_train_acc = abs(train_accuracy_score - test_accuracy_score)

            logging.info(f'Diff. test train accuracy: {diff_test_train_acc}')
            logging.info(f'Model accuracy: {model_accuracy}')

            mlflow.sklearn.log_model(sk_model=model,
                                     artifact_path="model_training",
                                     conda_env='conda.yaml',
                                     registered_model_name=model_name,
                                     input_example=X_train[:1, :],
                                     metadata=dict(stage="training",
                                                   index_number=index_number))
            
            if initial_base_score == 0:
                initial_base_score = base_score

            query = f"params.base_score ='{initial_base_score}'"
            all_experiments = [exp.experiment_id for exp in mlflow.search_experiments()]
            runs = mlflow.search_runs(
            experiment_ids=all_experiments,
            filter_string=query,
            run_view_type=ViewType.ACTIVE_ONLY,
            )

            if runs.empty:
                mlflow.log_params({"base_score": base_score, "threshold": threshold})

            mlflow.log_metric("train_f1_score", train_f1_score)
            mlflow.log_metric("test_f1_score", test_f1_score)
            mlflow.log_metric("train_f-beta_score", train_fbeta_score)
            mlflow.log_metric("test_f-beta_score", test_fbeta_score)
            mlflow.log_metric("train_roc_auc_score", train_roc_auc_score)
            mlflow.log_metric("test_roc_auc_score", test_roc_auc_score)
            mlflow.log_metric("train_precision_score", train_precision_score)
            mlflow.log_metric("test_precision_score", test_precision_score)
            mlflow.log_metric("train_recall_score", train_recall_score)
            mlflow.log_metric("test_recall_score", test_recall_score)
            mlflow.log_metric("model_accuracy", model_accuracy)
            mlflow.log_metric("train_accuracy_score", train_accuracy_score)
            mlflow.log_metric("test_accuracy_score", test_accuracy_score)

            # defining threshold
            if train_fbeta_score >= base_score and diff_test_train_acc < threshold:
                base_score = train_fbeta_score
                metric_info_artifact = MetricInfoArtifact(
                    model_name=model_name,
                    model_object=model,
                    train_f1_score=train_f1_score,
                    test_f1_score=test_f1_score,
                    train_fbeta_score=train_fbeta_score,
                    test_fbeta_score=test_fbeta_score,
                    train_roc_auc_score=train_roc_auc_score,
                    test_roc_auc_score=test_roc_auc_score,
                    train_precision_score=train_precision_score,
                    test_precision_score=test_precision_score,
                    train_recall_score=train_recall_score,
                    test_recall_score=test_recall_score,
                    model_accuracy=model_accuracy,
                    train_accuracy_score=train_accuracy_score,
                    test_accuracy_score=test_accuracy_score,
                    model_index=index_number
                )

                logging.info(f'Acceptable model found {metric_info_artifact}')
            index_number += 1

        if metric_info_artifact is None:
            logging.info(
                f"No Models score > base_score: f-beta {train_fbeta_score, test_fbeta_score}")
            logging.info(
                f"Models score  diff_test_train_accuracy: {diff_test_train_acc}")

        return metric_info_artifact

    except Exception as e:
        raise FraudDetectionException(e, sys) from e


class ModelFactory:
    def __init__(self, model_config_path: str = None, ) -> None:
        try:
            self.config: dict = ModelFactory.read_params(model_config_path)
            self.grid_search_cv_module: str = self.config[FACTORY_GRID_SEARCH_KEY][FACTORY_MODULE_KEY]
            self.grid_search_class_name: str = self.config[FACTORY_GRID_SEARCH_KEY][FACTORY_CLASS_KEY]
            self.grid_search_property_data: dict = dict(self.config[FACTORY_GRID_SEARCH_KEY][FACTORY_PARAMS_KEY])
            self.models_initialized_model_config: dict = dict(self.config[FACTORY_MODEL_SELECTION_KEY])
            self.initialized_model_list = None
            self.grid_search_best_model_list = None
        except Exception as e:
            raise FraudDetectionException(e, sys) from e

    @staticmethod
    def read_params(config_path: str) -> dict:
        try:
            with open(config_path) as yaml_file:
                config: dict = yaml.safe_load(yaml_file)
            return config
        except Exception as e:
            raise FraudDetectionException(e, sys)

    @staticmethod
    def class_for_name(module_name: str, class_name: str):
        try:
            # load the module, will raise ImportError if module cannot be loaded
            module = importlib.import_module(module_name)
            # get the class, will raise attribute error if class cannot be found
            logging.info(f'Executing command: from {module} import {class_name}')
            class_ref = getattr(module, class_name)
            return class_ref
        except Exception as e:
            raise FraudDetectionException(e, sys) from e

    @staticmethod
    def update_property_of_class(instance_ref: object, property_data: dict):
        try:
            if not isinstance(property_data, dict):
                raise Exception("property data parameter required to be a dictionary")
            for key, value in property_data.items():
                logging.info(f'Executing : ${str(instance_ref)}.{key}={value}')
                setattr(instance_ref, key, value)
            return instance_ref
        except Exception as e:
            raise FraudDetectionException(e, sys) from e

    def execute_grid_search_operation(self, initialized_model: InitializedModelDetails, input_feature,
                                      output_feature) -> GridSearchedBestModel:
        """
        execute_grid_search_operation(): function will perform parameter search operation, and
        it will return you the best optimistic  model with the best parameter:
        estimator: Model object
        param_grid: dictionary of parameter to perform search operation
        input_feature: all input features
        output_feature: Target/Dependent features
        ================================================================================
        return: Function will return GridSearchOperation object
        """
        try:
            # instantiating GridSearch CV
            grid_search_cv_ref = ModelFactory.class_for_name(module_name=self.grid_search_cv_module,
                                                             class_name=self.grid_search_class_name)
            grid_search_cv = grid_search_cv_ref(estimator=initialized_model.model,
                                                param_grid=initialized_model.param_grid_search)

            scoring = {
                'f1': make_scorer(f1_score),
                'precision': make_scorer(precision_score),
            }

            self.grid_search_property_data.update({"scoring": scoring,
                                                   'refit': 'f1'})

            grid_search_cv = ModelFactory.update_property_of_class(grid_search_cv, self.grid_search_property_data)

            message = f'\n{">>" * 20} Training {type(initialized_model.model).__name__} Started {"<<" * 20}\n'
            logging.info(message)

            grid_search_cv.fit(input_feature, output_feature)

            grid_search_best_model = GridSearchedBestModel(model_serial_number=initialized_model.model_serial_number,
                                                           model=initialized_model.model,
                                                           best_model=grid_search_cv.best_estimator_,
                                                           best_parameters=grid_search_cv.best_params_,
                                                           best_score=grid_search_cv.best_score_
                                                           )
            return grid_search_best_model

        except Exception as e:
            raise FraudDetectionException(e, sys) from e

    def get_initialized_model_list(self) -> List[InitializedModelDetails]:

        try:
            initialized_model_list = []
            for model_serial_number in self.models_initialized_model_config.keys():
                model_initialization_config = self.models_initialized_model_config[model_serial_number]
                model_obj_ref = ModelFactory.class_for_name(
                    module_name=model_initialization_config[FACTORY_MODULE_KEY],
                    class_name=model_initialization_config[FACTORY_CLASS_KEY])

                model = model_obj_ref()

                if FACTORY_PARAMS_KEY in model_initialization_config:
                    model_obj_property_data = dict(model_initialization_config[FACTORY_PARAMS_KEY])
                    model = ModelFactory.update_property_of_class(instance_ref=model,
                                                                  property_data=model_obj_property_data)
                param_grid_search = model_initialization_config[FACTORY_SEARCH_PARAM_GRID_KEY]
                model_name = f"{model_initialization_config[FACTORY_MODULE_KEY]}.{model_initialization_config[FACTORY_CLASS_KEY]}"

                model_initialization_config = InitializedModelDetails(
                    model_serial_number=model_serial_number,
                    model=model,
                    param_grid_search=param_grid_search,
                    model_name=model_name,
                )
                initialized_model_list.append(model_initialization_config)
            self.initialized_model_list = initialized_model_list
            return self.initialized_model_list
        except Exception as e:
            raise FraudDetectionException from e

    def initiate_best_parameter_search_for_initialized_models(self,
                                                              initialized_model_list: List[InitializedModelDetails],
                                                              input_feature,
                                                              output_feature) -> List[GridSearchedBestModel]:
        """
        initiate_best_parameter_search_for_initialized_models() :function will perform parameter search operation for
        each model in list, it will return you the best optimistic  model with the best parameter:
        estimator: Model object
        param_grid: dictionary of parameter to perform search operation
        input_feature: all input features
        output_feature: Target/Dependent features
        ================================================================================
        return: Function will return a GridSearchOperation for each model
        """
        try:
            self.grid_search_best_model_list = []
            for initialized_model in initialized_model_list:
                grid_searched_best_model = self.execute_grid_search_operation(
                    initialized_model=initialized_model,
                    input_feature=input_feature,
                    output_feature=output_feature
                )

                self.grid_search_best_model_list.append(grid_searched_best_model)
            return self.grid_search_best_model_list
        except Exception as e:
            raise FraudDetectionException(e, sys) from e

    @staticmethod
    def get_best_model_from_grid_searched_best_model_list(grid_searched_best_model_list: List[GridSearchedBestModel],
                                                          base_score=0.6) -> BestModel:
        """
        function Accepts the models with score greater than base accuracy threshold. 
        raise exception if no model accuracy greater than base accuracy found.
        args: 
            grid_searched_best_model_list: list of Grid-searched best models
            base_score: accuracy threshold set
        returns:
            best models with accuracy
        """
        try:
            best_model = None
            for grid_searched_best_model in grid_searched_best_model_list:
                if base_score < grid_searched_best_model.best_score:
                    logging.info(f"Acceptable model found: {grid_searched_best_model}")
                    base_score = grid_searched_best_model.best_score
                    best_model = grid_searched_best_model

                if not best_model:
                    raise Exception(f"None of Model has base Accuracy: {base_score}")

                logging.info(f'Best model: {best_model}')
                return best_model
        except Exception as e:
            raise FraudDetectionException(e, sys) from e

    def get_best_model(self, X, y, base_score) -> BestModel:
        try:

            logging.info("Started Initializing model from config file")
            initialized_model_list: List[InitializedModelDetails] = self.get_initialized_model_list()
            logging.info(f"initialized model : {initialized_model_list}")

            grid_searched_best_model_list: List[
                GridSearchedBestModel] = self.initiate_best_parameter_search_for_initialized_models(
                initialized_model_list=initialized_model_list,
                input_feature=X,
                output_feature=y,
            )
            return ModelFactory.get_best_model_from_grid_searched_best_model_list(
                grid_searched_best_model_list=grid_searched_best_model_list,
                base_score=base_score)
        except Exception as e:
            raise FraudDetectionException(e, sys) from e
