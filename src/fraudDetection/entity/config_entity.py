from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    source_url: str
    raw_data_dir: Path
    zip_dir: Path
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
    mlflow_uri: str


@dataclass(frozen=True)
class ModelPusherConfig:
    model_export_dir: Path
    saved_models_directory: str

@dataclass(frozen=True)
class TrainingPipelineConfig:
    artifacts_root: Path
    training_pipeline_name: str
