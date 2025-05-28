import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_metrics(y_true, y_pred):
    """
    Calculate various regression metrics.

    Args:
        y_true: True target values
        y_pred: Predicted target values

    Returns:
        dict: Dictionary containing various metrics
    """
    metrics = {
        "mse": mean_squared_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }
    return metrics


def plot_residuals(y_true, y_pred, title="Residual Plot"):
    """
    Plot residuals to check for patterns.

    Args:
        y_true: True target values
        y_pred: Predicted target values
        title: Plot title
    """
    residuals = y_true - y_pred

    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals)
    plt.axhline(y=0, color="r", linestyle="--")
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title(title)
    plt.savefig("residuals_plot.png")
    plt.close()


def plot_prediction_vs_actual(y_true, y_pred, title="Predicted vs Actual"):
    """
    Plot predicted vs actual values.

    Args:
        y_true: True target values
        y_pred: Predicted target values
        title: Plot title
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title(title)
    plt.savefig("prediction_vs_actual.png")
    plt.close()


def plot_feature_importance(model, feature_names):
    """
    Plot feature importance for tree-based models.

    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names
    """
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1]

    plt.figure(figsize=(12, 6))
    plt.title("Feature Importance")
    plt.bar(range(len(importance)), importance[indices])
    plt.xticks(range(len(importance)), [feature_names[i] for i in indices], rotation=45)
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    plt.close()
