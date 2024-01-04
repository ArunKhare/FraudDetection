"""
This module defines data classes for various artifacts used in the project.

Classes:
    - DataIngestionArtifact: Artifact representing the result of data ingestion.
    - DataValidationArtifact: Artifact representing the result of data validation.
    - DataTransformationArtifact: Artifact representing the result of data transformation.
    - ModelTrainerArtifact: Artifact representing the result of model training.
    - ModelEvaluationArtifact: Artifact representing the result of model evaluation.
    - ModelPusherArtifacts: Artifacts representing the result of model pushing.
    - Experiment: Class representing an experiment with associated metadata.

Attributes:
    - train_file_path (Path): Path to the training file.
    - test_file_path (Path): Path to the test file.
    - is_ingested (bool): Flag indicating whether data is ingested.
    - message (str): Additional message related to the artifact.

    # (similar pattern for attributes in other classes)

Examples:
    - data_ingestion_artifact = DataIngestionArtifact( train_file_path=Path("/path/to/train.csv"),
                                                test_file_path=Path("/path/to/test.csv"), is_ingested=True,
                                                message="Data successfully ingested.")
    - data_validation_artifact = DataValidationArtifact(report_file_path=Path("/path/to/validation_report.txt"),
                                                schema_file_path=Path("/path/to/schema.json"),
                                                class_proportion_train=0.8, is_validated=True,
                                                message="Data validation completed.")
    - data_transformation_artifact = DataTransformationArtifact(is_transformed=True, message="Data successfully transformed.",
                                                transformed_train_file_path=Path("/path/to/transformed_train.csv"),
                                                transformed_test_file_path=Path("/path/to/transformed_test.csv"),
                                                processed_object_file_path=Path("/path/to/processed_object.pkl"),
                                                impute_sampler_object_file_path=Path("/path/to/impute_sampler_object.pkl"))

    - model_trainer_artifact = ModelTrainerArtifact(is_trained=True,message="Model training completed successfully.",
                                                trained_model_file_path=Path("/path/to/trained_model.pkl"), train_accuracy=0.85,
                                                test_accuracy=0.82, train_f1_score=0.78,test_f1_score=0.75,
                                                 # (Include other attributes similarly))
    
    - model_evaluation_artifact = ModelEvaluationArtifact(is_model_accepted=True, 
                                                evaluated_model_path=Path("/path/to/evaluated_model.pkl"))
    simliar for other classes
"""
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
