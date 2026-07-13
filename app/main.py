import os
import sys
import json
from datetime import datetime
from contextlib import asynccontextmanager
import pandas as pd
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import ValidationError

from app.schemas import PredictionInput
from src.logger import logger
from src.exception import CustomException
from src.pipeline.prediction_pipeline import PredictionPipeline
from src.utils.db_helper import DatabaseHelper
from src.utils import read_yaml_file

# Globals to load pipeline objects
prediction_pipeline = None
db_helper = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Asynchronous lifespan manager that initializes the prediction pipeline and database helper
    on application startup and cleans up resources on shutdown.
    """
    global prediction_pipeline, db_helper
    try:
        logger.info("Initializing prediction pipeline and SQLite backend...")
        prediction_pipeline = PredictionPipeline()
        db_helper = DatabaseHelper()
        logger.info("API services successfully initialized within lifespan context.")
        yield
    except Exception as e:
        logger.error(f"Critical error on startup initialization: {str(e)}")
        raise e

app = FastAPI(
    title="Breast Cancer Prediction System",
    description="Enterprise MLOps Logistic Regression prediction service",
    version="1.0.0",
    lifespan=lifespan
)

# Ensure directories exist
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# Mount static asset and template path engines
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/reports", StaticFiles(directory="reports"), name="reports")
templates = Jinja2Templates(directory="templates")

# --- UI Routes ---

@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    try:
        logger.info("Serving root landing page request.")
        return templates.TemplateResponse(request, "index.html")
    except Exception as e:
        logger.error(f"Error serving landing page: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.get("/about", response_class=HTMLResponse)
async def read_about(request: Request):
    try:
        logger.info("Serving about page request.")
        return templates.TemplateResponse(request, "about.html")
    except Exception as e:
        logger.error(f"Error serving about page: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.get("/predict", response_class=HTMLResponse)
async def get_predict_form(request: Request):
    try:
        logger.info("Serving prediction query form request.")
        schema_path = prediction_pipeline.schema_path
        schema_data = read_yaml_file(str(schema_path))
        features = [col for col in schema_data["columns"].keys() if col != "target"]
        
        # Load sample input to guide user entries
        sample_input = PredictionInput.model_config["json_schema_extra"]["example"]
        
        # Load multiple test samples for random cycling
        test_samples = []
        test_csv_path = os.path.join("artifacts", "data_ingestion", "test_data.csv")
        if os.path.exists(test_csv_path):
            try:
                test_df = pd.read_csv(test_csv_path)
                # Sample 15 random rows
                sampled_df = test_df.sample(min(15, len(test_df)), random_state=42)
                # Convert to list of dicts without target
                for _, row in sampled_df.iterrows():
                    sample_dict = row.to_dict()
                    target_val = sample_dict.pop("target", None)
                    test_samples.append({
                        "features": sample_dict,
                        "target": int(target_val) if target_val is not None else None
                    })
            except Exception as read_err:
                logger.warning(f"Could not load test samples for page context: {read_err}")
                
        return templates.TemplateResponse(request, "predict.html", {
            "features": features,
            "sample_input": sample_input,
            "test_samples": test_samples,
            "form_data": {}
        })
    except Exception as e:
        logger.error(f"Error serving prediction form: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.post("/predict", response_class=HTMLResponse)
async def predict_tumor(request: Request):
    try:
        logger.info("Received HTTP POST inference request.")
        form_data = await request.form()
        form_dict = dict(form_data)
        
        # Coerce values to floats if possible
        processed_inputs = {}
        for k, v in form_dict.items():
            try:
                processed_inputs[k] = float(v)
            except ValueError:
                processed_inputs[k] = v
                
        # Validate with Pydantic schema
        try:
            input_validated = PredictionInput(**processed_inputs)
        except ValidationError as val_err:
            logger.warning(f"Inference parameters failed validation checks: {val_err.errors()}")
            
            schema_path = prediction_pipeline.schema_path
            schema_data = read_yaml_file(str(schema_path))
            features = [col for col in schema_data["columns"].keys() if col != "target"]
            sample_input = PredictionInput.model_config["json_schema_extra"]["example"]
            
            # Re-render input page highlighting field errors
            return templates.TemplateResponse(request, "predict.html", {
                "errors": val_err.errors(),
                "form_data": form_dict,
                "features": features,
                "sample_input": sample_input
            })
            
        # Create input DataFrame (use V2 model_dump method)
        input_df = pd.DataFrame([input_validated.model_dump()])
        
        # Run inference through PredictionPipeline
        pred_class, pred_label, confidence = prediction_pipeline.predict(input_df)
        
        # Write record to SQLite history log
        db_helper.insert_prediction(
            feature_values=input_validated.model_dump(),
            pred_class=pred_class,
            pred_label=pred_label,
            confidence=confidence
        )
        
        return templates.TemplateResponse(request, "result.html", {
            "prediction": pred_label,
            "confidence": f"{confidence * 100:.2f}%",
            "probability": f"{confidence:.4f}",
            "class": pred_class,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    except Exception as e:
        logger.error(f"Error handling prediction transaction: {str(e)}")
        ce = CustomException(e, sys)
        return templates.TemplateResponse(request, "error.html", {
            "error_code": 500,
            "error_msg": str(ce)
        }, status_code=500)

@app.get("/metrics", response_class=HTMLResponse)
async def read_metrics(request: Request):
    try:
        logger.info("Serving model metrics page.")
        # Load latest model performance report
        eval_report_path = os.path.join("artifacts", "model_evaluation", "evaluation_report.json")
        metrics_data = {}
        if os.path.exists(eval_report_path):
            with open(eval_report_path, "r") as f:
                metrics_data = json.load(f)
                
        # Load latest predictions history
        history = db_helper.fetch_all_predictions() if db_helper else []
        
        return templates.TemplateResponse(request, "metrics.html", {
            "metrics": metrics_data,
            "total_predictions": len(history),
            "history": history[:15] # Display top 15 records
        })
    except Exception as e:
        logger.error(f"Error serving metrics page: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

# --- API Routes ---

@app.get("/health")
async def health_check():
    """
    Standard service health probe returning connection states and file models check status.
    """
    try:
        logger.info("Health probe request received.")
        
        db_status = "disconnected"
        if db_helper:
            try:
                conn = db_helper._get_connection()
                conn.cursor().execute("SELECT 1")
                conn.close()
                db_status = "connected"
            except Exception:
                db_status = "error"
                
        status_info = {
            "status": "healthy",
            "model_loaded": prediction_pipeline is not None and prediction_pipeline.model is not None,
            "scaler_loaded": prediction_pipeline is not None and prediction_pipeline.scaler is not None,
            "database_connection": db_status
        }
        return JSONResponse(content=status_info, status_code=status.HTTP_200_OK)
    except Exception as e:
        logger.error(f"Health check diagnostics failed: {str(e)}")
        return JSONResponse(
            content={"status": "unhealthy", "error": str(e)},
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

# --- Global HTML Exception Templates ---

@app.exception_handler(HTTPException)
async def custom_http_exception_handler(request: Request, exc: HTTPException):
    logger.warning(f"HTTP custom exception thrown: {exc.status_code} - {exc.detail}")
    return templates.TemplateResponse(request, "error.html", {
        "error_code": exc.status_code,
        "error_msg": exc.detail
    }, status_code=exc.status_code)

@app.exception_handler(404)
async def custom_404_handler(request: Request, exc: Exception):
    logger.warning(f"404 error page served for path: {request.url.path}")
    return templates.TemplateResponse(request, "error.html", {
        "error_code": 404,
        "error_msg": "The page you are looking for does not exist."
    }, status_code=404)
