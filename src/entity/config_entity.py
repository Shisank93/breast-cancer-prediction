from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class TrainingPipelineConfig:
    artifacts_dir: Path

@dataclass(frozen=True)
class DataIngestionConfig:
    data_ingestion_dir: Path
    raw_data_file_path: Path
    train_data_file_path: Path
    test_data_file_path: Path
    test_size: float

@dataclass(frozen=True)
class DataValidationConfig:
    data_validation_dir: Path
    status_file_path: Path
    report_file_path: Path
    schema_file_path: Path

@dataclass(frozen=True)
class DataTransformationConfig:
    data_transformation_dir: Path
    transformed_train_file_path: Path
    transformed_test_file_path: Path
    scaler_file_path: Path

@dataclass(frozen=True)
class ModelTrainerConfig:
    model_trainer_dir: Path
    model_file_path: Path
    c_param: float
    max_iter: int
    solver: str
    penalty: str

@dataclass(frozen=True)
class ModelEvaluationConfig:
    model_evaluation_dir: Path
    report_file_path: Path
    model_file_path: Path
    scaler_file_path: Path
    test_data_file_path: Path
