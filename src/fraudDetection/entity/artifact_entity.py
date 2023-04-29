from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataIngestionArtifact:
    train_file_path: Path
    test_file_path: Path
    is_ingested: str
    message: str

@dataclass
class DataValidationArtifact:
    report_file_path: Path
    schema_file_path: Path
    is_validated: str
    message: str

@dataclass
class DataTransformationArtifact:
    is_transformed: str
    message: str 
    transformed_train_file_path: Path
    transformed_test_file_path: Path
    processed_object_file_path: Path

@dataclass
class ModelTrainerArtifact:
    is_trained: str
    message: str
    trained_model_file_path: Path
    precision: float
    recall: float
    f1_score: float
    train_accuracy: float
    test_accuracy:float
    model_accuracy:float

@dataclass
class ModelEvaluationArtifact:
    is_model_accepted: str
    evaluated_model_path: Path

@dataclass
class ModelPusherArtifacts:
    is_model_pusher: str
    export_model_file_path: Path

