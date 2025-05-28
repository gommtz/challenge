from fastapi import FastAPI, File, UploadFile, HTTPException
import pandas as pd
import numpy as np
import mlflow
from src.data.data_processor import DataProcessor
from src.models.model_trainer import ModelTrainer
import os

app = FastAPI(title="ML Challenge API")
data_processor = DataProcessor()
model_trainer = ModelTrainer()

# Global variable to store the model
model = None


@app.on_event("startup")
async def load_model():
    """Load the model at startup."""
    global model
    try:
        # Set MLflow tracking URI
        mlflow.set_tracking_uri("http://localhost:5000")

        # Get the latest run
        client = mlflow.tracking.MlflowClient()
        runs = client.search_runs(
            experiment_ids=["0"],  # Default experiment
            order_by=["start_time DESC"],
            max_results=1,
        )

        if not runs:
            raise Exception("No MLflow runs found. Please train the model first.")

        latest_run = runs[0]
        model_uri = f"runs:/{latest_run.info.run_id}/model"

        # Load the model
        model = mlflow.xgboost.load_model(model_uri)
        print(f"Successfully loaded model from run: {latest_run.info.run_id}")

    except Exception as e:
        print(f"Error loading model: {str(e)}")
        print("Please make sure to:")
        print("1. Start the MLflow server: mlflow server --host 0.0.0.0 --port 5000")
        print("2. Train the model first using the training script")
        raise e


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Make predictions on uploaded data."""
    if model is None:
        raise HTTPException(
            status_code=500, detail="Model not loaded. Please check the server logs."
        )

    try:
        # Read the uploaded file
        df = pd.read_csv(file.file)

        # Preprocess the data
        X = data_processor.preprocess_data(df, is_training=False)

        # Make predictions
        predictions = model.predict(X)

        # Create response
        response = pd.DataFrame({"target_pred": predictions})

        return response.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_exists": os.path.exists(data_processor.scaler_path),
    }
