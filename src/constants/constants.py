import os

# Config and schema file paths
CONFIG_FILE_PATH: str = os.path.join("config", "config.yaml")
SCHEMA_FILE_PATH: str = os.path.join("config", "schema.yaml")
DATABASE_NAME: str = "prediction_history.db"

# Model-related constants
TARGET_COLUMN: str = "target"
MODEL_FILE_NAME: str = "model.pkl"
SCALER_FILE_NAME: str = "scaler.pkl"

# Pipeline constants
PIPELINE_NAME: str = "breast_cancer_pipeline"
ARTIFACT_DIR: str = "artifacts"
