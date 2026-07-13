import os
import sys
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from src.logger import logger
from src.exception import CustomException
from src.entity import DataIngestionConfig, DataIngestionArtifact

class DataIngestion:
    """
    DataIngestion class loads the dataset, converts columns to snake_case, 
    saves raw data and splits train/test subsets.
    """
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
            logger.info("Initialized DataIngestion component.")
        except Exception as e:
            raise CustomException(e, sys) from e

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        logger.info("Entered initiate_data_ingestion method of DataIngestion class.")
        try:
            # Load Breast Cancer dataset from sklearn
            logger.info("Loading breast cancer dataset from sklearn...")
            cancer_data = load_breast_cancer()
            
            X = cancer_data.data
            y = cancer_data.target
            feature_names = cancer_data.feature_names
            
            # Map features to snake_case (e.g. 'mean radius' -> 'mean_radius')
            snake_feature_names = [name.replace(" ", "_") for name in feature_names]
            
            # Create DataFrame
            df = pd.DataFrame(data=X, columns=snake_feature_names)
            df["target"] = y
            
            logger.info(f"Dataset shape loaded: {df.shape}")
            
            # Save raw dataset
            raw_file_path = self.data_ingestion_config.raw_data_file_path
            logger.info(f"Saving raw dataset to {raw_file_path}...")
            os.makedirs(os.path.dirname(raw_file_path), exist_ok=True)
            df.to_csv(raw_file_path, index=False, header=True)
            
            # Split train & test sets (stratified to preserve class ratio)
            logger.info(f"Splitting dataset with test size {self.data_ingestion_config.test_size}...")
            train_df, test_df = train_test_split(
                df, 
                test_size=self.data_ingestion_config.test_size, 
                random_state=42, 
                stratify=df["target"]
            )
            
            # Save train and test sets
            train_file_path = self.data_ingestion_config.train_data_file_path
            test_file_path = self.data_ingestion_config.test_data_file_path
            
            logger.info(f"Saving train dataset (shape: {train_df.shape}) to {train_file_path}...")
            train_df.to_csv(train_file_path, index=False, header=True)
            
            logger.info(f"Saving test dataset (shape: {test_df.shape}) to {test_file_path}...")
            test_df.to_csv(test_file_path, index=False, header=True)
            
            data_ingestion_artifact = DataIngestionArtifact(
                raw_data_file_path=raw_file_path,
                train_data_file_path=train_file_path,
                test_data_file_path=test_file_path
            )
            logger.info(f"Data ingestion completed. Artifact created: {data_ingestion_artifact}")
            return data_ingestion_artifact
            
        except Exception as e:
            raise CustomException(e, sys) from e
