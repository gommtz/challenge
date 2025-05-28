import xgboost as xgb
import mlflow
import mlflow.xgboost
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np


class ModelTrainer:
    def __init__(self):
        self.model = None

    def train_model(self, X_train, y_train, X_val, y_val):
        """Train XGBoost model with MLflow tracking."""
        with mlflow.start_run():
            # Set model parameters
            params = {
                "objective": "reg:squarederror",
                "learning_rate": 0.1,
                "max_depth": 6,
                "min_child_weight": 1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "n_estimators": 100,
                "random_state": 42,
            }

            # Log parameters
            mlflow.log_params(params)

            # Train model
            self.model = xgb.XGBRegressor(**params)
            self.model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=10,
                verbose=False,
            )

            # Make predictions
            y_pred = self.model.predict(X_val)

            # Calculate metrics
            mse = mean_squared_error(y_val, y_pred)
            r2 = r2_score(y_val, y_pred)

            # Log metrics
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("r2", r2)

            # Log model
            mlflow.xgboost.log_model(self.model, "model")

            return mse, r2

    def predict(self, X):
        """Make predictions using the trained model."""
        return self.model.predict(X)

    def save_predictions(self, predictions, output_path):
        """Save predictions to CSV file."""
        np.savetxt(
            output_path, predictions, delimiter=",", header="target_pred", comments=""
        )
