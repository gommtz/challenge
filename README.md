
## Setup and Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

1. Start MLflow tracking server:
```bash
mlflow server --host 0.0.0.0 --port 5000
```

2. Run the training script:
```bash
python -m src.models.model_trainer
```

### Running the API

1. Build and run the Docker container:
```bash
docker build -t ml-challenge .
docker run -p 8000:8000 ml-challenge
```

2. Or run directly with uvicorn:
```bash
uvicorn src.api.main:app --reload
```

## API Endpoints

- POST `/predict`: Upload a CSV file to get predictions
- GET `/health`: Health check endpoint

## Model Evolution Strategy

1. **Feature Engineering**:
   - Implement feature selection using SHAP values
   - Create interaction features
   - Add polynomial features for non-linear relationships

2. **Model Improvements**:
   - Implement cross-validation
   - Try ensemble methods (Stacking, Blending)
   - Hyperparameter optimization using Optuna

3. **Monitoring and Maintenance**:
   - Implement model performance monitoring
   - Set up automated retraining pipeline
   - Add data drift detection

4. **Deployment Enhancements**:
   - Add authentication to API
   - Implement rate limiting
   - Add request validation
   - Set up CI/CD pipeline