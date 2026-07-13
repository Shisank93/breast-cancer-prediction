import sys
from fastapi.testclient import TestClient
from app.main import app
from src.logger import logger
from src.exception import CustomException
from app.schemas import PredictionInput

def verify_backend():
    """
    Executes in-memory integration tests against FastAPI backend endpoints.
    Ensures correct routing, template context rendering, validation error catches, and health logs.
    """
    try:
        logger.info("Initializing TestClient for FastAPI App...")
        with TestClient(app) as client:
            # 1. Health Probe
            logger.info("Testing /health API endpoint...")
            response = client.get("/health")
            assert response.status_code == 200, f"Expected 200, got {response.status_code}"
            health_json = response.json()
            assert health_json["status"] == "healthy"
            assert health_json["model_loaded"] is True
            assert health_json["scaler_loaded"] is True
            print("Health endpoint verified successfully!")
            
            # 2. GET landing page
            logger.info("Testing GET / endpoint...")
            response = client.get("/")
            assert response.status_code == 200
            assert "OncoPredict" in response.text
            print("Landing page GET verified successfully!")
            
            # 3. GET predict form
            logger.info("Testing GET /predict endpoint...")
            response = client.get("/predict")
            assert response.status_code == 200
            assert "Tumor Classification Tool" in response.text
            print("Prediction Form GET verified successfully!")
            
            # 4. POST predict (valid input)
            logger.info("Testing POST /predict with VALID data...")
            sample_data = PredictionInput.Config.json_schema_extra["example"]
            
            response = client.post("/predict", data=sample_data)
            assert response.status_code == 200
            assert "Classification:" in response.text
            assert "Benign" in response.text or "Malignant" in response.text
            print("Prediction POST (valid) verified successfully!")
            
            # 5. POST predict (invalid input range check)
            logger.info("Testing POST /predict with INVALID data (out of bounds)...")
            invalid_data = sample_data.copy()
            invalid_data["mean_radius"] = -5.0 # Should fail gt=0 check
            
            response = client.post("/predict", data=invalid_data)
            assert response.status_code == 200 # Catch and re-render form
            assert "Validation Errors" in response.text
            print("Validation error rendering verified successfully!")
            
            # 6. GET metrics
            logger.info("Testing GET /metrics...")
            response = client.get("/metrics")
            assert response.status_code == 200
            assert "Performance Dashboard" in response.text
            print("Metrics page GET verified successfully!")
            
        print("All Backend Endpoints and Validations verified successfully in Phase 4!")
        
    except Exception as e:
        logger.exception("Error during backend verification.")
        raise CustomException(e, sys) from e

if __name__ == "__main__":
    verify_backend()
