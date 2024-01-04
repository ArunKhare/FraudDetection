"""This module initialize the models from config.yaml to the instance of sklearn models
It provides functionalities to evaluate the trained models on various metrics and select the best model based on specified criteria.

Module Functions:
    - evaluate_classification_model: Evaluates the models on metrics like F1 score, precision, recall, ROC-AUC, etc.
    - ModelFactory: Class for cross-validating models and obtaining the best model configurations from config.yaml.
    - class_for_name: Instantiates sklearn models from the classes specified in the config file.
    - update_property_of_class: Updates the parameters of the sklearn class.
    - execute_grid_search_operation: Performs parameter search operation using GridSearchCV.
    - get_initialized_model_list: Retrieves a list of all initialized models from the configuration file.
    - initiate_best_parameter_search_for_initialized_models: Initiates the best parameter search for each initialized model.
    - get_best_model_from_grid_searched_best_model_list: Selects the best model from the list of GridSearchedBestModel instances.
    - get_best_model: Initializes models from the configuration file, performs grid search, and selects the best model.

Classes:
    - InitializedModelDetails: Represents details of an initialized model.
    - GridSearchedBestModel: Represents the best model obtained from grid search.
    - BestModel: Represents the final best model selected based on specified criteria.
    - MetricInfoArtifact: Represents a dictionary of evaluated metrics as an artifact.

Note: valid config 'model.yaml'_ within the config directory fraudDetection package,.
"""

import sys
from typing import List
import importlib
import numpy as np
import yaml
from sklearn.metrics import (
    f1_score,
    fbeta_score,
    roc_auc_score,
    precision_recall_curve,
    recall_score,
    classification_report,
    precision_score,
    accuracy_score,
    make_scorer,
)
from fraudDetection.exception import FraudDetectionException
from fraudDetection.logger import logging

from fraudDetection.constants import (
    FACTORY_CLASS_KEY,
    FACTORY_GRID_SEARCH_KEY,
    FACTORY_MODEL_SELECTION_KEY,
    FACTORY_MODULE_KEY,
    FACTORY_SEARCH_PARAM_GRID_KEY,
    FACTORY_PARAMS_KEY,
)

from fraudDetection.entity import (
    InitializedModelDetails,
    GridSearchedBestModel,
    BestModel,
    MetricInfoArtifact,
)


def evaluate_classification_model(
    model_list: list,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    base_score: float = 0.6,
    threshold: float = 0.05,
) -> MetricInfoArtifact:
    """Module level function to evaluate the models on the basis of metrics
    Args:
        model_list (List(str): list of trained models to be evaluated
        X_train (obj:'Numpy.ndarray'): array of training dataset
        y_train (obj:'Numpy.ndarray'): array of training target feature
        X_test: (obj:'Numpy.ndarray'): array of test dataset
        y_test  (obj:'Numpy.ndarray'): array of test target feature
        base_score (float = 0.6): base scire of metric to be evaluated
        threshold (float = 0.05): diff threshold of test and train  metrics
    Returns:
        metric_info_artifact (obj:'MetricInfoArtifact'): dict of evaluted matrics as artifact
    """
    try:
        index_number = 0
        metric_info_artifact = None
        # initial_base_score = 0
        for model in model_list:
            model_name = str(model)
            logging.info(
                f"\n{'>>' * 20} Starting evaluating model:{type(model).__name__}{'<<' * 20} "
            )

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
            logging.info(
                f"F-beta Score: train {train_fbeta_score}, test {test_fbeta_score}"
            )
            logging.info(
                f"Roc_Auc Score: train {train_roc_auc_score}, test {test_roc_auc_score}"
            )
            logging.info(
                f"Precision Score :  train {train_precision_score} test {test_precision_score}"
            )
            logging.info(
                f"Recall Score : train {train_recall_score} test {test_recall_score}"
            )
            logging.info(
                f"Classification Report : train \n {train_classification_report} test \n {test_classification_report}"
            )
            logging.info(
                f"Precision Recall Curve: train {train_precision_recall_curve}, test {test_precision_recall_curve}"
            )
            logging.info(
                f"Accuracy Score: train  {train_accuracy_score} test {test_recall_score}"
            )

            # calculating harmonic mean of train and test accuracy_score
            model_accuracy = (2 * train_accuracy_score * test_accuracy_score) / (
                train_accuracy_score + test_accuracy_score
            )
            diff_test_train_acc = abs(train_accuracy_score - test_accuracy_score)

            logging.info(f"Diff. test train accuracy: {diff_test_train_acc}")
            logging.info(f"Model accuracy: {model_accuracy}")

            # defining threshold
            if train_fbeta_score >= base_score and diff_test_train_acc < threshold:
                # base_score = train_fbeta_score
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
                    model_index=index_number,
                )

                logging.info(f"Acceptable model found {metric_info_artifact}")
            index_number += 1

        if metric_info_artifact is None:
            logging.info(
                f"No Models score > base_score: f-beta {train_fbeta_score, test_fbeta_score}"
            )
            logging.info(
                f"Models score  diff_test_train_accuracy: {diff_test_train_acc}"
            )

        return metric_info_artifact

    except Exception as e:
        raise FraudDetectionException(e, sys) from e


class ModelFactory:
    """cross validating the models and getting the best model configurations for models in config.yaml"""

    def __init__(
        self,
        model_config_path: str = None,
    ) -> None:
        """initialized ModelFactory object
        Args:
            mode_config_path (str): to store path to config.yaml
        """
        try:
            self.config: dict = ModelFactory.read_params(model_config_path)
            self.grid_search_cv_module: str = self.config[FACTORY_GRID_SEARCH_KEY][
                FACTORY_MODULE_KEY
            ]
            self.grid_search_class_name: str = self.config[FACTORY_GRID_SEARCH_KEY][
                FACTORY_CLASS_KEY
            ]
            self.grid_search_property_data: dict = dict(
                self.config[FACTORY_GRID_SEARCH_KEY][FACTORY_PARAMS_KEY]
            )
            self.models_initialized_model_config: dict = dict(
                self.config[FACTORY_MODEL_SELECTION_KEY]
            )
            self.initialized_model_list = None
            self.grid_search_best_model_list = None
        except Exception as e:
            raise FraudDetectionException(e, sys) from e

    @staticmethod
    def read_params(config_path: str) -> dict:
        """Read Parameters form model.yaml file
        Args:
            config_path (str): Path to model config file
        Returns:
            config  (Dict): parameters for model tunning
        """
        try:
            with open(config_path) as yaml_file:
                config: dict = yaml.safe_load(yaml_file)
            return config
        except Exception as e:
            raise FraudDetectionException(e, sys)

    @staticmethod
    def class_for_name(module_name: str, class_name: str):
        """instantiate the sklearn model from models in model config file
        Args:
            module_name (str): module name of sklearn model  mention in config file
            class_name (str): model class name
        Returns:
            class_ref (obj:'sklearn'): referenc to the model class
        """
        try:
            # load the module, will raise ImportError if module cannot be loaded
            module = importlib.import_module(module_name)
            # get the class, will raise attribute error if class cannot be found
            logging.info(f"Executing command: from {module} import {class_name}")
            class_ref = getattr(module, class_name)
            return class_ref
        except Exception as e:
            raise FraudDetectionException(e, sys) from e

    @staticmethod
    def update_property_of_class(instance_ref: object, property_data: dict):
        """update the parameters of the sklearn class
        Args:
            instance_ref (obj:'sklearn'): instance of class
            property_data (dict): parameters of class
        Returns:
            instance_ref (obj:'sklearn): instance with parameters updated
        """
        try:
            if not isinstance(property_data, dict):
                raise Exception("property data parameter required to be a dictionary")
            for key, value in property_data.items():
                logging.info(f"Executing : ${str(instance_ref)}.{key}={value}")
                setattr(instance_ref, key, value)
            return instance_ref
        except Exception as e:
            raise FraudDetectionException(e, sys) from e

    def execute_grid_search_operation(
        self, initialized_model: InitializedModelDetails, input_feature, output_feature
    ) -> GridSearchedBestModel:
        """
        function will perform parameter search operation, and it will return you the best optimistic  model with the
        best parameter:
        Args:
            input_feature (obj:'pd.DataFrame'):  all input features
            output_featur (obj:'pd.Series'): Target/Dependent features
            initalized_model(obj:'InitializedModelsDetails'): initialized model from the list
        Returns:
            grid_search_best_model (obj:'GridSearchedBestModel': contains 'best (model,parameters,score)'
        """
        try:
            # instantiating GridSearch CV
            grid_search_cv_ref = ModelFactory.class_for_name(
                module_name=self.grid_search_cv_module,
                class_name=self.grid_search_class_name,
            )
            grid_search_cv = grid_search_cv_ref(
                estimator=initialized_model.model,
                param_grid=initialized_model.param_grid_search,
            )

            scoring = {
                "f1": make_scorer(f1_score),
                "precision": make_scorer(precision_score),
            }

            self.grid_search_property_data.update({"scoring": scoring, "refit": "f1"})

            grid_search_cv = ModelFactory.update_property_of_class(
                grid_search_cv, self.grid_search_property_data
            )

            message = f'\n{">>" * 20} Training {type(initialized_model.model).__name__} Started {"<<" * 20}\n'
            logging.info(message)

            grid_search_cv.fit(input_feature, output_feature)

            grid_search_best_model = GridSearchedBestModel(
                model_serial_number=initialized_model.model_serial_number,
                model=initialized_model.model,
                best_model=grid_search_cv.best_estimator_,
                best_parameters=grid_search_cv.best_params_,
                best_score=grid_search_cv.best_score_,
            )
            return grid_search_best_model

        except Exception as e:
            raise FraudDetectionException(e, sys) from e

    def get_initialized_model_list(self) -> List[InitializedModelDetails]:
        """get the list of  all the initialized model
        Returns:
           model_initialization_config (obj:'InitializedModelDetails'): list of initialized model, params, name
        """
        try:
            initialized_model_list = []
            for model_serial_number in self.models_initialized_model_config.keys():
                model_initialization_config = self.models_initialized_model_config[
                    model_serial_number
                ]
                model_obj_ref = ModelFactory.class_for_name(
                    module_name=model_initialization_config[FACTORY_MODULE_KEY],
                    class_name=model_initialization_config[FACTORY_CLASS_KEY],
                )

                model = model_obj_ref()

                if FACTORY_PARAMS_KEY in model_initialization_config:
                    model_obj_property_data = dict(
                        model_initialization_config[FACTORY_PARAMS_KEY]
                    )
                    model = ModelFactory.update_property_of_class(
                        instance_ref=model, property_data=model_obj_property_data
                    )
                param_grid_search = model_initialization_config[
                    FACTORY_SEARCH_PARAM_GRID_KEY
                ]
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

    def initiate_best_parameter_search_for_initialized_models(
        self,
        initialized_model_list: List[InitializedModelDetails],
        input_feature,
        output_feature,
    ) -> List[GridSearchedBestModel]:
        """
        function will perform parameter search operation for
        each model in list, it will return you the best optimistic  model with the
        Args:
            initialized_model_list (obj:List(InitializedModelDetails)): list of initialized models
            input_feature (obj:'pd.DataFrame'): input data features
            Output_feature (obj:'pd.DataFrame'): output data features
        Returns:
            grid_search_best_model_list(obj:'List(GridSearchedBestModel)'): contains
            best parameter:
            estimator: Model object
            param_grid: dictionary of parameter to perform search operation
        """
        try:
            self.grid_search_best_model_list = []
            for initialized_model in initialized_model_list:
                grid_searched_best_model = self.execute_grid_search_operation(
                    initialized_model=initialized_model,
                    input_feature=input_feature,
                    output_feature=output_feature,
                )

                self.grid_search_best_model_list.append(grid_searched_best_model)
            return self.grid_search_best_model_list
        except Exception as e:
            raise FraudDetectionException(e, sys) from e

    @staticmethod
    def get_best_model_from_grid_searched_best_model_list(
        grid_searched_best_model_list: List[GridSearchedBestModel], base_score=0.6
    ) -> BestModel:
        """
        function Accepts the models with score greater than base accuracy threshold.
        raise exception if no model accuracy greater than base accuracy found.
        Args:
            grid_searched_best_model_list (obj:'List(GridSearchedBestModel)'): list of Grid-searched best models
            base_score (float): accuracy threshold set
        Returns:
            best models (obj:'sklearn') : model with best fbeta and accuracy score
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

                logging.info(f"Best model: {best_model}")
                return best_model
        except Exception as e:
            raise FraudDetectionException(e, sys) from e

    def get_best_model(self, X, y, base_score) -> BestModel:
        """implement the Grid search
        Args:
            X (obj:'pd.DataFrame'): input data
            y (obj:'pd.DataFrame'): target input
        Return:
            object (obj:'sklearn'): best model
        """
        try:
            logging.info("Started Initializing model from config file")
            initialized_model_list: List[
                InitializedModelDetails
            ] = self.get_initialized_model_list()
            logging.info(f"initialized model : {initialized_model_list}")

            grid_searched_best_model_list: List[
                GridSearchedBestModel
            ] = self.initiate_best_parameter_search_for_initialized_models(
                initialized_model_list=initialized_model_list,
                input_feature=X,
                output_feature=y,
            )
            return ModelFactory.get_best_model_from_grid_searched_best_model_list(
                grid_searched_best_model_list=grid_searched_best_model_list,
                base_score=base_score,
            )
        except Exception as e:
            raise FraudDetectionException(e, sys) from e
