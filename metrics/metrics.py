import numpy as np

# Model Performance Metrics
# Used for cross-model benchmarking: lower values indicate higher predictive precision.

def rmse(y_true, y_pred):
    """
    Root Mean Square Error (RMSE)
    Measures the average magnitude of error, with a higher penalty for large outliers.
    """
    return np.sqrt(((y_true - y_pred) ** 2).mean())

def mae(y_true, y_pred):
    """
    Mean Absolute Error (MAE)
    Measures the average magnitude of error in a linear scale, providing a more
    robust metric against significant outliers.
    """
    return abs(y_true - y_pred).mean()