import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os


class DataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.scaler_path = "src/data/models/scaler.joblib"

    def load_data(self, file_path):
        """Load data from CSV file."""
        return pd.read_csv(file_path)

    def preprocess_data(self, df, is_training=True):
        """Preprocess the data for model training or prediction, including outlier capping and scaling."""
        outlier_limits_path = "src/data/models/outlier_limits.joblib"
        if is_training:
            X = df.drop("target", axis=1)
            y = df["target"]

            # Calculate IQR-based outlier caps for each feature
            Q1 = X.quantile(0.25)
            Q3 = X.quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outlier_limits = {col: (lower[col], upper[col]) for col in X.columns}

            # Cap outliers
            X_capped = X.copy()
            for col in X.columns:
                X_capped[col] = X_capped[col].clip(
                    lower=outlier_limits[col][0], upper=outlier_limits[col][1]
                )

            # Save outlier limits
            os.makedirs(os.path.dirname(outlier_limits_path), exist_ok=True)
            joblib.dump(outlier_limits, outlier_limits_path)

            # Scale features
            X_scaled = self.scaler.fit_transform(X_capped)

            # Save the fitted scaler
            os.makedirs(os.path.dirname(self.scaler_path), exist_ok=True)
            joblib.dump(self.scaler, self.scaler_path)

            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )

            return X_train, X_val, y_train, y_val
        else:
            X = df
            # Load the fitted scaler
            if os.path.exists(self.scaler_path):
                self.scaler = joblib.load(self.scaler_path)
            else:
                raise Exception("Scaler not found. Please train the model first.")
            # Load outlier limits
            if os.path.exists(outlier_limits_path):
                outlier_limits = joblib.load(outlier_limits_path)
            else:
                raise Exception(
                    "Outlier limits not found. Please train the model first."
                )
            # Cap outliers using training limits
            X_capped = X.copy()
            for col in X.columns:
                if col in outlier_limits:
                    X_capped[col] = X_capped[col].clip(
                        lower=outlier_limits[col][0], upper=outlier_limits[col][1]
                    )
            X_scaled = self.scaler.transform(X_capped)
            return X_scaled
