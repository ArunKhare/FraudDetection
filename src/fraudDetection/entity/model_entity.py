from dataclasses import dataclass

@dataclass
class InitializedModelDetails:
    model_serial_no : int
    model : str
    param_grid_search : str
    model_name : str

@dataclass
class GridSearchBestModel:
    model_serial_number : int
    model : str
    best_model : str
    best_parameter : str
    best_score : float

@dataclass
class BestModel:
    model_serial_number :int
    model : str
    best_model : str
    best_parameters: str
    best_score : float

@dataclass
class MetricInfoArtifact:
    model_name : str
    model_object : str
    train_f1_score : float
    test_f1_score: float
    train_precision_score : float
    train_recall_score : float
    train_label_rank_avg_precision_curve : float
    model_accuracy: float
    model_index :int
    