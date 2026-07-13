import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from src.logger import logger
from src.exception import CustomException
from src.entity import DataTransformationConfig, DataTransformationArtifact, DataIngestionArtifact, DataValidationArtifact
from src.utils import save_object, save_numpy_array_data

class DataTransformation:
    """
    DataTransformation class loads datasets, scales features using StandardScaler,
    combines inputs and target labels, and serializes the scaler and binary datasets.
    """
    def __init__(self, data_transformation_config: DataTransformationConfig, 
                 data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_artifact: DataValidationArtifact):
        try:
            self.data_transformation_config = data_transformation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_artifact = data_validation_artifact
            logger.info("Initialized DataTransformation component.")
        except Exception as e:
            raise CustomException(e, sys) from e

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Loads split datasets, checks verification validation state, runs scaling,
        and saves numpy data outputs and scaler serialization pkl.
        """
        logger.info("Entered initiate_data_transformation method of DataTransformation class.")
        try:
            # Safety Check: Verify validation status from the validation artifact
            validation_status = self.data_validation_artifact.validation_status
            if not validation_status:
                raise Exception("Data Validation failed! Cannot proceed with Data Transformation.")
                
            # Read train/test CSV data
            train_df = pd.read_csv(self.data_ingestion_artifact.train_data_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_data_file_path)
            
            logger.info(f"Loaded train data shape: {train_df.shape}, test data shape: {test_df.shape}")
            
            # Separate features and target
            target_column = "target"
            
            X_train = train_df.drop(columns=[target_column])
            y_train = train_df[target_column]
            
            X_test = test_df.drop(columns=[target_column])
            y_test = test_df[target_column]
            
            logger.info("Initializing StandardScaler...")
            scaler = StandardScaler()
            
            # Fit and transform features
            logger.info("Fitting and transforming train features...")
            X_train_scaled = scaler.fit_transform(X_train)
            
            logger.info("Transforming test features...")
            X_test_scaled = scaler.transform(X_test)
            
            # Concatenate features and target arrays
            train_arr = np.c_[X_train_scaled, np.array(y_train)]
            test_arr = np.c_[X_test_scaled, np.array(y_test)]
            
            # Output paths
            train_npy_path = self.data_transformation_config.transformed_train_file_path
            test_npy_path = self.data_transformation_config.transformed_test_file_path
            scaler_path = self.data_transformation_config.scaler_file_path
            
            # Save numpy datasets
            logger.info(f"Saving transformed train array to {train_npy_path}...")
            save_numpy_array_data(str(train_npy_path), train_arr)
            
            logger.info(f"Saving transformed test array to {test_npy_path}...")
            save_numpy_array_data(str(test_npy_path), test_arr)
            
            # Save scaler object
            logger.info(f"Saving scaler object to {scaler_path}...")
            save_object(str(scaler_path), scaler)
            
            artifact = DataTransformationArtifact(
                transformed_train_file_path=train_npy_path,
                transformed_test_file_path=test_npy_path,
                scaler_file_path=scaler_path
            )
            logger.info(f"Data transformation completed. Artifact created: {artifact}")
            return artifact
            
        except Exception as e:
            raise CustomException(e, sys) from e
