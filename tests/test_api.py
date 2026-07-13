import os
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.schemas import PredictionInput

client = TestClient(app)

def test_health_endpoint():
    """
    Asserts health diagnostics return 200 OK and show models loaded successfully.
    """
    with client:
        response = client.get("/health")
        assert response.status_code == 200
        health = response.json()
        assert health["status"] == "healthy"
        assert health["model_loaded"] is True
        assert health["scaler_loaded"] is True

def test_pages_endpoints():
    """
    Asserts visual landing pages render and serve with correct header strings.
    """
    with client:
        # Test index page
        response = client.get("/")
        assert response.status_code == 200
        assert "OncoPredict" in response.text
        
        # Test predict page form
        response = client.get("/predict")
        assert response.status_code == 200
        assert "Tumor Classification Tool" in response.text
        
        # Test about page
        response = client.get("/about")
        assert response.status_code == 200
        assert "FNA Biopsy" in response.text

def test_prediction_flow():
    """
    Asserts standard valid predictions insert correctly and invalid bounds trigger validation.
    """
    with client:
        sample_data = PredictionInput.model_config["json_schema_extra"]["example"]
        
        # Post valid form data
        response = client.post("/predict", data=sample_data)
        assert response.status_code == 200
        assert "Classification:" in response.text
        
        # Post invalid data (negative radius)
        invalid_data = sample_data.copy()
        invalid_data["mean_radius"] = -2.0
        response = client.post("/predict", data=invalid_data)
        assert response.status_code == 200
        assert "Validation Errors" in response.text
