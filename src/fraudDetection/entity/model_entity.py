"""Model Details and Metrics Artifacts Module

This module defines data classes representing details and metrics related to machine learning models.

Classes:
    - InitializedModelDetails: Represents details about the initialization of a machine learning model.
    - GridSearchedBestModel: Represents the best model selected after grid search.
    - BestModel: Represents the best model and its details.
    - MetricInfoArtifact: Represents various metrics related to a machine learning model.

Attributes:
    - model_serial_number (str): Serial number uniquely identifying a machine learning model.
    - model (str): Type or name of the machine learning model.
    - param_grid_search (str): Details about the parameter grid search.
    - model_name (str): Name or identifier for the model.
    - best_model (str): Type or name of the best machine learning model.
    - best_parameters (str): Best hyperparameters for the selected model.
    - best_score (float): Best score achieved by the model.
    - model_object (str): Object representing the machine learning model.
    - train_f1_score (float): F1 score on the training set.
    - test_f1_score (float): F1 score on the test set.
    - train_fbeta_score (float): F-beta score on the training set.
    - test_fbeta_score (float): F-beta score on the test set.
    - train_roc_auc_score (float): ROC AUC score on the training set.
    - test_roc_auc_score (float): ROC AUC score on the test set.
    - train_precision_score (float): Precision score on the training set.
    - test_precision_score (float): Precision score on the test set.
    - train_recall_score (float): Recall score on the training set.
    - test_recall_score (float): Recall score on the test set.
    - model_accuracy (float): Overall accuracy of the model.
    - train_accuracy_score (float): Accuracy score on the training set.
    - test_accuracy_score (float): Accuracy score on the test set.
    - model_index (int): Index or identifier for the model.

Examples:
    - initialized_model = InitializedModelDetails(model_serial_number="123", model="RandomForest",
                                                param_grid_search="{'n_estimators': [50, 100], 'max_depth': [None, 10]}",
                                                 model_name="RF_Model")
    - best_model = BestModel(model_serial_number="456", model="LogisticRegression", best_model="LogReg",
                             best_parameters="{'C': 0.1, 'penalty': 'l2'}", best_score=0.85)
    - metric_info = MetricInfoArtifact(model_name="SVM_Model", model_object="SVMClassifier()",
                                       train_f1_score=0.78, test_f1_score=0.75, model_index=1)
"""
from dataclasses import dataclass


@dataclass
class InitializedModelDetails:
    model_serial_number: str
    model: str
    param_grid_search: str
    model_name: str


@dataclass
class GridSearchedBestModel:
    model_serial_number: str
    model: str
    best_model: str
    best_parameters: str
    best_score: float


@dataclass
class BestModel:
    model_serial_number: str
    model: str
    best_model: str
    best_parameters: str
    best_score: float


@dataclass
class MetricInfoArtifact:
    model_name: str
    model_object: str
    train_f1_score: float
    test_f1_score: float
    train_fbeta_score: float
    test_fbeta_score: float
    train_roc_auc_score: float
    test_roc_auc_score: float
    train_precision_score: float
    test_precision_score: float
    train_recall_score: float
    test_recall_score: float
    model_accuracy: float
    train_accuracy_score: float
    test_accuracy_score: float
    model_index: int
