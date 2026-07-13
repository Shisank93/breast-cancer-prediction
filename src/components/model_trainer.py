import os
import sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from src.logger import logger
from src.exception import CustomException
from src.entity import ModelTrainerConfig, ModelTrainerArtifact, DataTransformationArtifact
from src.utils import save_object, load_numpy_array_data

class ModelTrainer:
    """
    ModelTrainer class loads training data, extracts features and target,
    configures hyperparameters, fits LogisticRegression, and serializes the model.
    """
    def __init__(self, model_trainer_config: ModelTrainerConfig, data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
            logger.info("Initialized ModelTrainer component.")
        except Exception as e:
            raise CustomException(e, sys) from e

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        """
        Loads preprocessed numpy arrays, configures and trains LogisticRegression,
        saves the model object, and returns a ModelTrainerArtifact.
        """
        logger.info("Entered initiate_model_trainer method of ModelTrainer class.")
        try:
            # Load transformed train array
            train_npy_path = self.data_transformation_artifact.transformed_train_file_path
            logger.info(f"Loading transformed train array from {train_npy_path}...")
            train_arr = load_numpy_array_data(str(train_npy_path))
            
            # Separate X_train and y_train
            X_train = train_arr[:, :-1]
            y_train = train_arr[:, -1]
            
            logger.info(f"Train features shape: {X_train.shape}, target shape: {y_train.shape}")
            
            # Initialize Logistic Regression
            logger.info(
                f"Initializing Logistic Regression with hyperparameters: "
                f"C={self.model_trainer_config.c_param}, "
                f"max_iter={self.model_trainer_config.max_iter}, "
                f"solver={self.model_trainer_config.solver}, "
                f"penalty={self.model_trainer_config.penalty}"
            )
            
            lr_model = LogisticRegression(
                C=self.model_trainer_config.c_param,
                max_iter=self.model_trainer_config.max_iter,
                solver=self.model_trainer_config.solver,
                penalty=self.model_trainer_config.penalty,
                random_state=42
            )
            
            # Train model
            logger.info("Fitting Logistic Regression model...")
            lr_model.fit(X_train, y_train)
            
            # Save model
            model_path = self.model_trainer_config.model_file_path
            logger.info(f"Saving trained model to {model_path}...")
            save_object(str(model_path), lr_model)
            
            artifact = ModelTrainerArtifact(model_file_path=model_path)
            logger.info(f"Model training completed. Artifact created: {artifact}")
            return artifact
            
        except Exception as e:
            raise CustomException(e, sys) from e
