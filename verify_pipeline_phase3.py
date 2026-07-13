import os
import sys
import numpy as np
from src.config import ConfigurationManager
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation
from src.logger import logger
from src.exception import CustomException

def run_training_pipeline():
    """
    Executes the full pipeline stages: Ingestion -> Validation -> Transformation -> Training -> Evaluation.
    Ensures all outputs, reports, and visualization metrics are created correctly.
    """
    try:
        logger.info("Starting training pipeline verification (Phase 3)...")
        
        # 1. Config Manager
        config_manager = ConfigurationManager()
        
        # 2. Data Ingestion
        ingestion_config = config_manager.get_data_ingestion_config()
        ingestion = DataIngestion(data_ingestion_config=ingestion_config)
        ingestion_artifact = ingestion.initiate_data_ingestion()
        
        # 3. Data Validation
        validation_config = config_manager.get_data_validation_config()
        validation = DataValidation(
            data_validation_config=validation_config,
            data_ingestion_artifact=ingestion_artifact
        )
        validation_artifact = validation.initiate_data_validation()
        
        # 4. Data Transformation
        transformation_config = config_manager.get_data_transformation_config()
        transformation = DataTransformation(
            data_transformation_config=transformation_config,
            data_ingestion_artifact=ingestion_artifact,
            data_validation_artifact=validation_artifact
        )
        transformation_artifact = transformation.initiate_data_transformation()
        
        # 5. Model Training
        trainer_config = config_manager.get_model_trainer_config()
        trainer = ModelTrainer(
            model_trainer_config=trainer_config,
            data_transformation_artifact=transformation_artifact
        )
        trainer_artifact = trainer.initiate_model_trainer()
        
        assert os.path.exists(trainer_artifact.model_file_path), "Trained model file not found!"
        print("Model Trainer completed and verified successfully!")
        
        # 6. Model Evaluation
        eval_config = config_manager.get_model_evaluation_config()
        evaluation = ModelEvaluation(
            model_evaluation_config=eval_config,
            data_transformation_artifact=transformation_artifact,
            model_trainer_artifact=trainer_artifact
        )
        eval_artifact = evaluation.evaluate_model()
        
        assert os.path.exists(eval_artifact.report_file_path), "Evaluation report JSON file not found!"
        assert os.path.exists("reports/figures/confusion_matrix.png"), "Confusion matrix heatmap not found!"
        assert os.path.exists("reports/figures/roc_curve.png"), "ROC curve plot not found!"
        
        print(f"Model Acceptance Status: {eval_artifact.is_model_accepted}")
        print("Model Evaluation completed and verified successfully!")
        print("All Phase 3 Pipeline steps executed and verified successfully!")
        
    except Exception as e:
        logger.exception("Error in running training pipeline.")
        raise CustomException(e, sys) from e

if __name__ == "__main__":
    run_training_pipeline()
