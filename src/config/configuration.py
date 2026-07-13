import os
import sys
from pathlib import Path
from src.constants import CONFIG_FILE_PATH, SCHEMA_FILE_PATH
from src.entity import (
    TrainingPipelineConfig,
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig,
)
from src.utils import read_yaml_file
from src.logger import logger
from src.exception import CustomException

class ConfigurationManager:
    """
    ConfigurationManager parses yaml files and exposes configurations for all stages.
    """
    def __init__(self, config_filepath: str = CONFIG_FILE_PATH, schema_filepath: str = SCHEMA_FILE_PATH):
        try:
            self.config = read_yaml_file(config_filepath)
            self.schema_filepath = Path(schema_filepath)
            self.artifacts_dir = Path(self.config["artifacts_dir"])
            os.makedirs(self.artifacts_dir, exist_ok=True)
            logger.info(f"Initialized ConfigurationManager. Artifacts directory: {self.artifacts_dir}")
        except Exception as e:
            raise CustomException(e, sys) from e

    def get_training_pipeline_config(self) -> TrainingPipelineConfig:
        """
        Retrieves global training pipeline configuration.
        """
        try:
            return TrainingPipelineConfig(artifacts_dir=self.artifacts_dir)
        except Exception as e:
            raise CustomException(e, sys) from e

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """
        Retrieves configuration for the Data Ingestion component.
        """
        try:
            ingestion_config = self.config["data_ingestion"]
            ingestion_dir = self.artifacts_dir / ingestion_config["data_ingestion_dir"]
            os.makedirs(ingestion_dir, exist_ok=True)

            return DataIngestionConfig(
                data_ingestion_dir=ingestion_dir,
                raw_data_file_path=ingestion_dir / ingestion_config["raw_data_file"],
                train_data_file_path=ingestion_dir / ingestion_config["train_data_file"],
                test_data_file_path=ingestion_dir / ingestion_config["test_data_file"],
                test_size=float(ingestion_config["test_size"]),
            )
        except Exception as e:
            raise CustomException(e, sys) from e

    def get_data_validation_config(self) -> DataValidationConfig:
        """
        Retrieves configuration for the Data Validation component.
        """
        try:
            validation_config = self.config["data_validation"]
            validation_dir = self.artifacts_dir / validation_config["data_validation_dir"]
            os.makedirs(validation_dir, exist_ok=True)

            return DataValidationConfig(
                data_validation_dir=validation_dir,
                status_file_path=validation_dir / validation_config["status_file"],
                report_file_path=validation_dir / validation_config["report_file"],
                schema_file_path=self.schema_filepath,
            )
        except Exception as e:
            raise CustomException(e, sys) from e

    def get_data_transformation_config(self) -> DataTransformationConfig:
        """
        Retrieves configuration for the Data Transformation component.
        """
        try:
            transformation_config = self.config["data_transformation"]
            transformation_dir = self.artifacts_dir / transformation_config["data_transformation_dir"]
            os.makedirs(transformation_dir, exist_ok=True)

            return DataTransformationConfig(
                data_transformation_dir=transformation_dir,
                transformed_train_file_path=transformation_dir / transformation_config["transformed_train_file"],
                transformed_test_file_path=transformation_dir / transformation_config["transformed_test_file"],
                scaler_file_path=transformation_dir / transformation_config["scaler_file"],
            )
        except Exception as e:
            raise CustomException(e, sys) from e

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        """
        Retrieves configuration for the Model Trainer component.
        """
        try:
            trainer_config = self.config["model_trainer"]
            trainer_dir = self.artifacts_dir / trainer_config["model_trainer_dir"]
            os.makedirs(trainer_dir, exist_ok=True)

            return ModelTrainerConfig(
                model_trainer_dir=trainer_dir,
                model_file_path=trainer_dir / trainer_config["model_file"],
                c_param=float(trainer_config["c_param"]),
                max_iter=int(trainer_config["max_iter"]),
                solver=str(trainer_config["solver"]),
                penalty=str(trainer_config["penalty"]),
            )
        except Exception as e:
            raise CustomException(e, sys) from e

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        """
        Retrieves configuration for the Model Evaluation component.
        """
        try:
            eval_config = self.config["model_evaluation"]
            eval_dir = self.artifacts_dir / eval_config["model_evaluation_dir"]
            os.makedirs(eval_dir, exist_ok=True)

            ingestion_config = self.get_data_ingestion_config()
            transformation_config = self.get_data_transformation_config()
            trainer_config = self.get_model_trainer_config()

            return ModelEvaluationConfig(
                model_evaluation_dir=eval_dir,
                report_file_path=eval_dir / eval_config["report_file"],
                model_file_path=trainer_config.model_file_path,
                scaler_file_path=transformation_config.scaler_file_path,
                test_data_file_path=ingestion_config.test_data_file_path,
            )
        except Exception as e:
            raise CustomException(e, sys) from e
