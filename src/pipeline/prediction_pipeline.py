import sys
import pandas as pd
import numpy as np
from src.logger import logger
from src.exception import CustomException
from src.config import ConfigurationManager
from src.utils import load_object, read_yaml_file

class PredictionPipeline:
    """
    PredictionPipeline handles real-time inference by loading the serialized
    scaler and model, processing input features, and returning the outputs.
    """
    def __init__(self):
        try:
            logger.info("Initializing PredictionPipeline...")
            # Load configuration manager to fetch artifact paths dynamically
            config_manager = ConfigurationManager()
            trainer_config = config_manager.get_model_trainer_config()
            transformation_config = config_manager.get_data_transformation_config()
            
            self.model_path = trainer_config.model_file_path
            self.scaler_path = transformation_config.scaler_file_path
            self.schema_path = config_manager.schema_filepath
            
            logger.info(f"Loading scaler from: {self.scaler_path}")
            self.scaler = load_object(str(self.scaler_path))
            
            logger.info(f"Loading model from: {self.model_path}")
            self.model = load_object(str(self.model_path))
            
            logger.info("Scaler and Model loaded successfully into PredictionPipeline.")
        except Exception as e:
            raise CustomException(e, sys) from e

    def predict(self, features: pd.DataFrame) -> tuple:
        """
        Aligns columns, scales features, runs model, and returns:
        (prediction_class: int, prediction_label: str, confidence: float)
        """
        try:
            logger.info("PredictionPipeline: Initiating scaling and prediction...")
            
            # Read schema to align columns correctly
            schema_data = read_yaml_file(str(self.schema_path))
            feature_cols = [col for col in schema_data["columns"].keys() if col != "target"]
            
            # Align features according to expected order
            features = features[feature_cols]
            
            # Apply feature scaling
            scaled_features = self.scaler.transform(features)
            
            # Perform inference
            pred_class = int(self.model.predict(scaled_features)[0])
            probabilities = self.model.predict_proba(scaled_features)[0]
            confidence = float(probabilities[pred_class])
            
            # Map classes: 0 = Malignant, 1 = Benign
            pred_label = "Benign" if pred_class == 1 else "Malignant"
            
            logger.info(f"Prediction complete. Class: {pred_class} ({pred_label}), Confidence: {confidence:.4f}")
            return pred_class, pred_label, confidence
        except Exception as e:
            raise CustomException(e, sys) from e
