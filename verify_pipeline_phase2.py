import os
import sys
import numpy as np
from src.config import ConfigurationManager
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from notebooks.eda_script import generate_eda_report
from src.logger import logger
from src.exception import CustomException

def test_pipeline():
    """
    Executes all Phase 2 pipeline stages in order and asserts correct file creation.
    """
    try:
        logger.info("Starting Phase 2 Pipeline Verification...")
        
        # 1. Configuration Manager
        config_manager = ConfigurationManager()
        
        # 2. Data Ingestion
        ingestion_config = config_manager.get_data_ingestion_config()
        ingestion = DataIngestion(data_ingestion_config=ingestion_config)
        ingestion_artifact = ingestion.initiate_data_ingestion()
        
        # Verify Ingestion outputs
        assert os.path.exists(ingestion_artifact.raw_data_file_path), "Raw data file not found!"
        assert os.path.exists(ingestion_artifact.train_data_file_path), "Train data file not found!"
        assert os.path.exists(ingestion_artifact.test_data_file_path), "Test data file not found!"
        print("Data Ingestion completed and verified successfully!")
        
        # 3. Data Validation
        validation_config = config_manager.get_data_validation_config()
        validation = DataValidation(
            data_validation_config=validation_config,
            data_ingestion_artifact=ingestion_artifact
        )
        validation_artifact = validation.initiate_data_validation()
        
        # Verify Validation outputs
        assert os.path.exists(validation_artifact.status_file_path), "Validation status file not found!"
        assert os.path.exists(validation_artifact.report_file_path), "Validation report file not found!"
        with open(validation_artifact.status_file_path, "r") as f:
            status = f.read()
        assert status == "True", "Validation status is not True!"
        print("Data Validation completed and verified successfully!")
        
        # 4. Automated EDA Report
        print("Generating automated EDA report...")
        generate_eda_report(
            raw_data_path=str(ingestion_artifact.raw_data_file_path),
            report_dir="reports"
        )
        assert os.path.exists("reports/eda_report.md"), "EDA report markdown not found!"
        assert os.path.exists("reports/figures/class_distribution.png"), "Class distribution plot not found!"
        assert os.path.exists("reports/figures/correlation_heatmap.png"), "Correlation heatmap not found!"
        print("EDA generation completed and verified successfully!")
        
        # 5. Data Transformation
        transformation_config = config_manager.get_data_transformation_config()
        transformation = DataTransformation(
            data_transformation_config=transformation_config,
            data_ingestion_artifact=ingestion_artifact,
            data_validation_artifact=validation_artifact
        )
        transformation_artifact = transformation.initiate_data_transformation()
        
        # Verify Transformation outputs
        assert os.path.exists(transformation_artifact.transformed_train_file_path), "Transformed train array file not found!"
        assert os.path.exists(transformation_artifact.transformed_test_file_path), "Transformed test array file not found!"
        assert os.path.exists(transformation_artifact.scaler_file_path), "Scaler pickle file not found!"
        
        # Check integrity of saved numpy datasets
        train_arr = np.load(transformation_artifact.transformed_train_file_path)
        test_arr = np.load(transformation_artifact.transformed_test_file_path)
        print(f"Transformed train shape: {train_arr.shape}")
        print(f"Transformed test shape: {test_arr.shape}")
        
        print("Data Transformation completed and verified successfully!")
        print("All Phase 2 Pipeline steps executed and verified successfully!")
        
    except Exception as e:
        logger.exception("Error in pipeline run.")
        raise CustomException(e, sys) from e

if __name__ == "__main__":
    test_pipeline()
