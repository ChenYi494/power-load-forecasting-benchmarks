import os
import joblib
from sklearn.ensemble import RandomForestRegressor

class RFModel:
    def __init__(self, n_estimators=100, max_depth=None, min_samples_leaf=1, max_features="sqrt", random_seed=42):
        # Initialize the ensemble regressor
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,           # Total number of decision trees
            max_depth=max_depth,                 # Controls tree complexity and inference latency jitter
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,           # Feature subsetting for variance reduction
            random_state=random_seed,            # Fixed seed for experimental reproducibility
            n_jobs=1                             # Forced single-threading for deterministic benchmarking
        )

        # Store configuration for metadata logging and hyperparameter tracking
        self.params_cfg = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "max_features": max_features,
            "random_seed": random_seed
        }

    def train(self, X, y):
        """
        Fits the ensemble model to the training data.
        Note: RF does not require a validation set for Early Stopping.
        """
        # Ensure y is a 1D array as required by Scikit-Learn
        self.model.fit(X, y.ravel() if len(y.shape) > 1 else y)

    def predict(self, X):
        """Generates point-wise forecasts based on input features."""
        return self.model.predict(X)

    def save(self, path):
        """Serializes the ensemble to binary with Zlib compression to optimize storage footprint."""
        joblib.dump(self.model, path, compress=3)

    @staticmethod
    def model_size(path):
        """Quantifies the storage footprint of the saved ensemble in Megabytes (MB)."""
        if not os.path.exists(path): return 0
        return os.path.getsize(path) / (1024 * 1024)

    def complexity(self):
        """
        Estimates model complexity based on the total cumulative nodes across all estimators.
        This metric directly correlates with the memory consumption of the trained forest.
        """
        if hasattr(self.model, "estimators_"):
            # Aggregate total nodes across all individual decision trees
            n_nodes = sum(tree.tree_.node_count for tree in self.model.estimators_)
            if n_nodes < 1000:
                return "low"
            elif n_nodes < 10000:
                return "medium"
            else:
                return "high"
        return "low"

    def get_hyperparameters(self):
        """Retrieves the architectural parameters for experiment tracking."""
        return self.params_cfg