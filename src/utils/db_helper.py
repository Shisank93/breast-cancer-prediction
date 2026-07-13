import sqlite3
import os
import sys
from datetime import datetime
from src.logger import logger
from src.exception import CustomException
from src.constants import DATABASE_NAME, SCHEMA_FILE_PATH
from src.utils import read_yaml_file

class DatabaseHelper:
    """
    DatabaseHelper manages the SQLite database schema and handles insertions
    and retrievals of prediction request records.
    """
    def __init__(self, db_name: str = DATABASE_NAME):
        try:
            self.db_path = os.path.join(os.getcwd(), db_name)
            self.schema = read_yaml_file(SCHEMA_FILE_PATH)
            self.features = [col for col in self.schema["columns"].keys() if col != "target"]
            logger.info(f"Initialized DatabaseHelper with DB path: {self.db_path}")
            self._create_table()
        except Exception as e:
            raise CustomException(e, sys) from e

    def _get_connection(self) -> sqlite3.Connection:
        try:
            return sqlite3.connect(self.db_path)
        except Exception as e:
            raise CustomException(e, sys) from e

    def _create_table(self) -> None:
        """
        Dynamically builds the table schema based on structural features listed in schema.yaml.
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Construct column definitions for feature set
            cols_def = []
            for col in self.features:
                cols_def.append(f"{col} REAL")
                
            cols_sql = ", ".join(cols_def)
            create_table_query = f"""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                {cols_sql},
                prediction_class INTEGER,
                prediction_label TEXT,
                confidence REAL,
                timestamp TEXT
            )
            """
            cursor.execute(create_table_query)
            conn.commit()
            conn.close()
            logger.info("Checked and created predictions table if not exists.")
        except Exception as e:
            raise CustomException(e, sys) from e

    def insert_prediction(self, feature_values: dict, pred_class: int, pred_label: str, confidence: float) -> int:
        """
        Inserts feature inputs, predicted label, confidence probability and execution timestamp.
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Extract features in the precise schema ordering
            ordered_vals = []
            columns_list = []
            placeholders = []
            
            for col in self.features:
                val = feature_values.get(col)
                if val is None:
                    raise ValueError(f"Feature '{col}' value is missing from prediction payload.")
                ordered_vals.append(float(val))
                columns_list.append(col)
                placeholders.append("?")
                
            columns_list.extend(["prediction_class", "prediction_label", "confidence", "timestamp"])
            ordered_vals.extend([int(pred_class), str(pred_label), float(confidence), datetime.now().isoformat()])
            placeholders.extend(["?", "?", "?", "?"])
            
            insert_query = f"""
            INSERT INTO predictions ({", ".join(columns_list)})
            VALUES ({", ".join(placeholders)})
            """
            cursor.execute(insert_query, ordered_vals)
            conn.commit()
            last_row_id = cursor.lastrowid
            conn.close()
            logger.info(f"Inserted prediction record into database. Row ID: {last_row_id}")
            return last_row_id
        except Exception as e:
            raise CustomException(e, sys) from e

    def fetch_all_predictions(self) -> list:
        """
        Retrieves all historic records sorted chronologically descending.
        """
        try:
            conn = self._get_connection()
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM predictions ORDER BY id DESC")
            rows = cursor.fetchall()
            results = [dict(row) for row in rows]
            conn.close()
            logger.info(f"Fetched {len(results)} prediction logs from SQLite.")
            return results
        except Exception as e:
            raise CustomException(e, sys) from e
