import os
import sys
import numpy as np
from src.logger import logger
from src.exception import CustomException
from src.config import ConfigurationManager
from src.utils import (
    read_yaml_file,
    write_yaml_file,
    save_object,
    load_object,
    save_numpy_array_data,
    load_numpy_array_data,
)

def test_logging_and_exceptions():
    logger.info("Starting verification script...")
    try:
        logger.info("Attempting to trigger a CustomException...")
        1 / 0
    except Exception as e:
        logger.info("Caught exception. Formatting with CustomException...")
        ce = CustomException(e, sys)
        logger.info(f"CustomException string representation:\n{str(ce)}")
        print("Logging and custom exceptions verified successfully!")

def test_configuration_manager():
    logger.info("Verifying ConfigurationManager...")
    config_manager = ConfigurationManager()
    
    # Ingestion Config
    ingestion_config = config_manager.get_data_ingestion_config()
    logger.info(f"Ingestion raw path: {ingestion_config.raw_data_file_path}")
    assert ingestion_config.test_size == 0.2
    
    # Validation Config
    validation_config = config_manager.get_data_validation_config()
    logger.info(f"Validation status path: {validation_config.status_file_path}")
    
    # Transformation Config
    transformation_config = config_manager.get_data_transformation_config()
    logger.info(f"Transformation scaler path: {transformation_config.scaler_file_path}")
    
    # Trainer Config
    trainer_config = config_manager.get_model_trainer_config()
    logger.info(f"Trainer model file path: {trainer_config.model_file_path}")
    
    # Evaluation Config
    eval_config = config_manager.get_model_evaluation_config()
    logger.info(f"Evaluation report path: {eval_config.report_file_path}")
    
    print("Configuration Manager verified successfully!")

def test_utils():
    logger.info("Verifying shared utilities...")
    
    # Ensure artifacts folder exists for temp tests
    os.makedirs("artifacts", exist_ok=True)
    
    # Test yaml write/read
    temp_yaml = "artifacts/temp_test.yaml"
    test_dict = {"test_key": "test_value"}
    write_yaml_file(temp_yaml, test_dict, replace=True)
    loaded_dict = read_yaml_file(temp_yaml)
    assert loaded_dict == test_dict
    os.remove(temp_yaml)
    
    # Test serialization of numpy array
    temp_npy = "artifacts/temp_test.npy"
    test_arr = np.array([1.0, 2.0, 3.0])
    save_numpy_array_data(temp_npy, test_arr)
    loaded_arr = load_numpy_array_data(temp_npy)
    assert np.allclose(loaded_arr, test_arr)
    os.remove(temp_npy)
    
    # Test object serialization (dill)
    temp_pkl = "artifacts/temp_test.pkl"
    class DummyObj:
        def __init__(self):
            self.value = 42
    dummy = DummyObj()
    save_object(temp_pkl, dummy)
    loaded_dummy = load_object(temp_pkl)
    assert loaded_dummy.value == 42
    os.remove(temp_pkl)
    
    print("Utilities verified successfully!")

if __name__ == "__main__":
    print("Executing tests...")
    test_logging_and_exceptions()
    test_configuration_manager()
    test_utils()
    print("All Phase 1 foundational tests passed successfully!")
