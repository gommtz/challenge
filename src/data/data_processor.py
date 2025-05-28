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
        self.scaler_path = "models/scaler.joblib"

    def load_data(self, file_path):
        """Load data from CSV file."""
        return pd.read_csv(file_path)

    def perform_eda(self, df):
        """Perform exploratory data analysis."""
        # Basic statistics
        print("Basic Statistics:")
        print(df.describe())

        # Check for missing values
        print("\nMissing Values:")
        print(df.isnull().sum())

        # Correlation analysis
        plt.figure(figsize=(12, 8))
        correlation_matrix = df.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", center=0)
        plt.title("Feature Correlation Matrix")
        plt.tight_layout()
        plt.savefig("correlation_matrix.png")

        # Distribution of target variable
        plt.figure(figsize=(10, 6))
        sns.histplot(df["target"], kde=True)
        plt.title("Target Variable Distribution")
        plt.savefig("target_distribution.png")

        # Feature distributions
        plt.figure(figsize=(15, 10))
        for i, feature in enumerate(df.columns[:-1], 1):
            plt.subplot(4, 5, i)
            sns.histplot(df[feature], kde=True)
            plt.title(f"Feature {i-1}")
        plt.tight_layout()
        plt.savefig("feature_distributions.png")

    def preprocess_data(self, df, is_training=True):
        """Preprocess the data for model training or prediction."""
        if is_training:
            X = df.drop("target", axis=1)
            y = df["target"]

            # Scale features
            X_scaled = self.scaler.fit_transform(X)

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

            X_scaled = self.scaler.transform(X)
            return X_scaled
