from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    source_url: str
    raw_data_dir: Path
    unzip_dir: Path
    ingested_dir: Path
    ingested_train_dir: Path
    ingested_test_dir: Path

@dataclass(frozen=True)
class DataValidationConfig:
    schema_dir: Path
    schema_file_name: str
    report_file_name: str

@dataclass(frozen=True)
class DataTransformationConfig:
    tranformed_dir:Path
    tranformed_train_dir: Path
    transformed_test_dir: Path
    preprocessing_dir: Path
    preprocessed_object_file_name: str

@dataclass(frozen=True)
class ModelTrainerConfig:
    trained_model_file_path:Path
    base_accuracy: float
    model_config_file_path: str

@dataclass(frozen=True)
class ModelEvaluationConfig:
    model_evaluation_file_name: str
    time_stamp: str
    mlflow_uri: str
    
@dataclass(frozen=True)
class ModelPusherConfig:
    model_export_dir: str

@dataclass(frozen=True)
class TraningPipelineConfig:
    artifacts_root: Path