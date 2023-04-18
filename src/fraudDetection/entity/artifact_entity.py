from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataIngestionArtifact:
    train_file_path: Path
    test_file_path: Path
    is_ingested: str
    message: str
class DataValidationArtifact:
    report_filename = Path



