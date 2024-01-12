"""This module defines data classes for various configuration settings used in the project, 
including paths and other configurations.

Classes:
    - DataIngestionConfig: Configuration for data ingestion process.
    - DataValidationConfig: Configuration for data validation process.
    - DataTransformationConfig: Configuration for data validation process.
    - ModelTrainerConfig: Configuration for model trainer process.
    - ModelEvaluationConfig: Configuration for model evaluation process.
    - ModelPusherConfig: Configuration for model pusher process.
    - TrainingPipelineConfig: Configuration for training pipeline process.

Attributes:
    - source_url (str): The source URL for data ingestion.
    - kaggle_config_file (Path): Path to the Kaggle configuration file.
    - zip_data_dir(Path): Path to the zip data directory
    - raw_data_dir (Path): Path to the raw data directory.
    - ingested_dir (Path): Path to the ingested data directory.
    - ingested_train_dir (Path): Path to the ingested training data directory.
    - ingested_test_dir (Path): Path to the ingested test data directory.
    - stratify (str): Stratification parameter for data splitting (default: None).
    - test_size (float): Test set size for data splitting (default: 0.2).

    - schema_file_name (str): Name of the schema file for data validation.
    - schema_file_path (Path): Path to the schema file.
    - report_file_name (str): Name of the report file for data validation.
    - report_file_path (Path): Path to the report file.

    - transformed_dir (Path): Path to the transformed data directory.
    - transformed_train_dir (Path): Path to the transformed training data directory.
    - transformed_test_dir (Path): Path to the transformed test data directory.
    - preprocessing_object_file_path (Path): Path to the preprocessing object file.
    - imputer_sampler_object_file_path (Path): Path to the imputer sampler object file.

    - trained_model_file_path (Path): Path to the trained model file.
    - base_score (float): Base score for the model.
    - model_config_file_path (str): Path to the model configuration file.
    - threshold_diff_train_test_acc (float): Threshold difference between train and test accuracy.

    - model_evaluation_file_path (Path): Path to the model evaluation file.
    - time_stamp (str): Timestamp for model evaluation.

    - model_export_dir (Path): Path to the model export directory.
    - saved_models_directory (str): Directory for saved models.

    - artifacts_root (Path): Root directory for artifacts.
    - training_pipeline_name (str): Name of the training pipeline.

Examples:
    - training_config = TrainingPipelineConfig(
        artifacts_root=Path("/path/to/artifacts"),
        training_pipeline_name="ExamplePipeline"
    )
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    source_url: str
    zip_data_dir: Path
    raw_data_dir: Path
    ingested_dir: Path
    ingested_train_dir: Path
    ingested_test_dir: Path
    stratify: str = None
    test_size: float = 0.2


@dataclass(frozen=True)
class DataValidationConfig:
    schema_file_name: str
    schema_file_path: Path
    report_file_name: str
    report_file_path: Path


@dataclass(frozen=True)
class DataTransformationConfig:
    transformed_dir: Path
    transformed_train_dir: Path
    transformed_test_dir: Path
    preprocessing_object_file_path: Path
    imputer_sampler_object_file_path: Path


@dataclass(frozen=True)
class ModelTrainerConfig:
    trained_model_file_path: Path
    base_score: float
    model_config_file_path: str
    threshold_diff_train_test_acc: float


@dataclass(frozen=True)
class ModelEvaluationConfig:
    model_evaluation_file_path: Path
    time_stamp: str


@dataclass(frozen=True)
class ModelPusherConfig:
    model_export_dir: Path
    saved_models_directory: str


@dataclass(frozen=True)
class TrainingPipelineConfig:
    artifacts_root: Path
    training_pipeline_name: str
