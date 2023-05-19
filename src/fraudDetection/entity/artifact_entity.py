from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, time


@dataclass
class DataIngestionArtifact:
    train_file_path: Path
    test_file_path: Path
    is_ingested: bool
    message: str


@dataclass
class DataValidationArtifact:
    report_file_path: Path
    schema_file_path: Path
    class_proportion_train: float
    is_validated: bool
    message: str


@dataclass
class DataTransformationArtifact:
    is_transformed: bool
    message: str
    transformed_train_file_path: Path
    transformed_test_file_path: Path
    processed_object_file_path: Path
    impute_sampler_object_file_path: Path


@dataclass
class ModelTrainerArtifact:
    is_trained: bool
    message: str
    trained_model_file_path: Path
    train_accuracy: float
    test_accuracy: float
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
    threshold: float


@dataclass
class ModelEvaluationArtifact:
    is_model_accepted: bool
    evaluated_model_path: Path


@dataclass
class ModelPusherArtifacts:
    is_model_pusher: bool
    export_model_file_path: Path


@dataclass
class Experiment:
    experiment_id: str
    initialization_timestamp: datetime
    artifact_timestamp: datetime
    running_status: str
    start_time: time
    stop_time: time
    execution_time: time
    message: str
    experiment_file_path: Path
    accuracy: float
    is_model_accepted: str
