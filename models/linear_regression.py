from sklearn.linear_model import LinearRegression, Ridge
import joblib
import os

class LinearModel:
    def __init__(self, reg_type='none', alpha=1.0):
        # Determine algorithm based on configuration (Ordinary Least Squares vs. Ridge)
        if reg_type == 'L2':
            # Ridge Regression mitigates overfitting by penalizing large coefficients (L2 penalty)
            self.model = Ridge(alpha=alpha)
        else:
            # Traditional OLS Linear Regression
            self.model = LinearRegression()

    def train(self, X, y):
        """
        Executes the training process by calculating optimal weights (w) and intercept (b)
        via the ordinary least squares or Ridge loss function.
        """
        self.model.fit(X, y)

    def predict(self, X):
        """Generates point-wise predictions using the learned coefficients."""
        return self.model.predict(X)

    def save(self, path):
        """Serializes the model to binary format with Zlib compression to optimize storage usage."""
        joblib.dump(self.model, path, compress=3)

    @staticmethod
    def model_size(path):
        """Quantifies the storage footprint of the saved model in Megabytes (MB)."""
        return os.path.getsize(path) / (1024 * 1024)

    def complexity(self):
        """
        Estimates model complexity based on the input feature dimensionality (degree of freedom).
        """
        n_features = getattr(self.model, "n_features_in_", None)
        if n_features is None:
            return "low"
        elif n_features < 10:
            return "low"
        elif n_features < 100:
            return "medium"
        else:
            return "high"

    def get_hyperparameters(self):
        """Retrieves the full configuration dictionary of the Scikit-Learn estimator."""
        return self.model.get_params()



