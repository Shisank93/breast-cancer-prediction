import os
import sys
import json
import pandas as pd
from src.logger import logger
from src.exception import CustomException
from src.entity import DataValidationConfig, DataValidationArtifact, DataIngestionArtifact
from src.utils import read_yaml_file

class DataValidation:
    """
    DataValidation class verifies column schema, sizes, types, target values,
    and missing record statistics. Writes logs, raw text status, and JSON report.
    """
    def __init__(self, data_validation_config: DataValidationConfig, data_ingestion_artifact: DataIngestionArtifact):
        try:
            self.data_validation_config = data_validation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            logger.info("Initialized DataValidation component.")
        except Exception as e:
            raise CustomException(e, sys) from e

    def validate_all_columns(self) -> bool:
        """
        Loads the dataset and performs schema, type, target range and null check validations.
        """
        try:
            # Load raw data
            raw_data_path = self.data_ingestion_artifact.raw_data_file_path
            logger.info(f"Loading raw data for validation from: {raw_data_path}")
            df = pd.read_csv(raw_data_path)
            
            # Read schema file
            schema_path = self.data_validation_config.schema_file_path
            logger.info(f"Reading schema file from: {schema_path}")
            schema = read_yaml_file(str(schema_path))
            schema_columns = schema["columns"]
            
            validation_status = True
            report = {
                "dataset_shape": list(df.shape),
                "columns_validation": {},
                "types_validation": {},
                "missing_values": df.isnull().sum().to_dict(),
                "summary": ""
            }
            
            # Column count check
            if len(df.columns) != len(schema_columns):
                validation_status = False
                logger.warning(f"Column count mismatch: Expected {len(schema_columns)}, got {len(df.columns)}")
                report["columns_validation"]["count_match"] = False
            else:
                report["columns_validation"]["count_match"] = True
                
            # Column existence and type checks
            for column, expected_type in schema_columns.items():
                if column not in df.columns:
                    validation_status = False
                    logger.warning(f"Column [{column}] is missing from dataset.")
                    report["columns_validation"][column] = "Missing"
                else:
                    report["columns_validation"][column] = "Present"
                    
                    # Check type compatibility
                    col_dtype = str(df[column].dtype)
                    if expected_type == "float" and not ("float" in col_dtype or "int" in col_dtype):
                        validation_status = False
                        logger.warning(f"Column [{column}] type mismatch: Expected float-compatible, got {col_dtype}")
                        report["types_validation"][column] = f"Mismatch (Expected float, got {col_dtype})"
                    elif expected_type == "int" and not ("int" in col_dtype or "bool" in col_dtype):
                        validation_status = False
                        logger.warning(f"Column [{column}] type mismatch: Expected integer-compatible, got {col_dtype}")
                        report["types_validation"][column] = f"Mismatch (Expected int, got {col_dtype})"
                    else:
                        report["types_validation"][column] = "OK"

            # Check if target values are binary
            if "target" in df.columns:
                unique_targets = set(df["target"].unique())
                if not unique_targets.issubset({0, 1}):
                    validation_status = False
                    logger.warning(f"Target column values are not binary. Found: {unique_targets}")
                    report["target_binary_check"] = f"Failed (Found values: {list(unique_targets)})"
                else:
                    report["target_binary_check"] = "OK"
            else:
                validation_status = False
                report["target_binary_check"] = "Failed (Target column missing)"
            
            report["validation_status"] = validation_status
            report["summary"] = "Data validation passed." if validation_status else "Data validation failed."
            
            # Save validation outputs
            status_file_path = self.data_validation_config.status_file_path
            report_file_path = self.data_validation_config.report_file_path
            
            logger.info(f"Writing validation status (value: {validation_status}) to {status_file_path}...")
            os.makedirs(os.path.dirname(status_file_path), exist_ok=True)
            with open(status_file_path, "w") as f:
                f.write(str(validation_status))
                
            logger.info(f"Writing validation report JSON to {report_file_path}...")
            with open(report_file_path, "w") as f:
                json.dump(report, f, indent=4)
                
            return validation_status
            
        except Exception as e:
            raise CustomException(e, sys) from e

    def initiate_data_validation(self) -> DataValidationArtifact:
        """
        Runs validation checks and returns DataValidationArtifact.
        """
        try:
            logger.info("Initiating data validation component...")
            validation_status = self.validate_all_columns()
            
            artifact = DataValidationArtifact(
                validation_status=validation_status,
                status_file_path=self.data_validation_config.status_file_path,
                report_file_path=self.data_validation_config.report_file_path
            )
            logger.info(f"Data validation completed. Artifact created: {artifact}")
            return artifact
        except Exception as e:
            raise CustomException(e, sys) from e
