import mlflow
from src.data.data_processor import DataProcessor
from src.models.model_trainer import ModelTrainer
import os


def main():
    # Set MLflow tracking URI
    mlflow.set_tracking_uri("http://localhost:5000")

    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)

    # Initialize components
    data_processor = DataProcessor()
    model_trainer = ModelTrainer()

    # Load and process data
    print("Loading training data...")
    train_data = data_processor.load_data("src/data/processed/training_data.csv")

    print("Performing EDA...")
    data_processor.perform_eda(train_data)

    print("Preprocessing data...")
    X_train, X_val, y_train, y_val = data_processor.preprocess_data(train_data)

    print("Training model...")
    mse, r2 = model_trainer.train_model(X_train, y_train, X_val, y_val)
    print(f"Training completed. MSE: {mse}, R2: {r2}")

    # Make predictions on blind test data
    print("Making predictions on blind test data...")
    blind_test_data = data_processor.load_data("src/data/processed/blind_test_data.csv")
    X_test = data_processor.preprocess_data(blind_test_data, is_training=False)
    predictions = model_trainer.predict(X_test)

    # Save predictions
    print("Saving predictions...")
    model_trainer.save_predictions(predictions, "src/data/processed/predictions.csv")
    print("Done!")


if __name__ == "__main__":
    main()
