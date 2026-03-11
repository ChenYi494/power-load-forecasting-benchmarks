import torch
import torch.nn as nn
import torch.utils.data as data
import os
import copy
import joblib

class MLPModel:
    def __init__(self, input_dim, hidden_layers, dropout, lr):
        # Build network architecture dynamically
        layers = []
        prev_dim = input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))

        self.model = nn.Sequential(*layers)
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # Store parameters for hyperparameter logging
        self.params_cfg = {
            "input_dim": input_dim,
            "hidden_layers": hidden_layers,
            "dropout": dropout,
            "lr": lr
        }

    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32, patience=5):
        """
        Executes the training loop with validation monitoring and Early Stopping.
        Returns the number of effective epochs completed.
        """
        # Convert to PyTorch Tensors
        X_t = torch.tensor(X_train, dtype=torch.float32)
        y_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        X_v = torch.tensor(X_val, dtype=torch.float32)
        y_v = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

        dataset = data.TensorDataset(X_t, y_t)
        loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

        best_val_loss = float('inf')
        epochs_no_improve = 0
        effective_epochs = 0

        # Snapshot of initial state for safety
        self.best_state = copy.deepcopy(self.model.state_dict())

        for epoch in range(epochs):
            self.model.train()
            for xb, yb in loader:
                pred = self.model(xb)
                loss = self.loss_fn(pred, yb)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            effective_epochs += 1

            # Validation / Performance Monitoring
            self.model.eval()
            with torch.no_grad():
                val_pred = self.model(X_v)
                val_loss = self.loss_fn(val_pred, y_v).item()

            # Early Stopping Logic: Restore best weights if no improvement is seen
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                self.best_state = self.model.state_dict()
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                self.model.load_state_dict(self.best_state)  # Rollback to optimal weights
                break

        return effective_epochs

    def predict(self, X):
        """Generates point-wise forecasts based on input features."""
        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            return self.model(X_t).cpu().numpy().flatten()

    def save(self, path):
        """Serializes the model to binary with Zlib compression to optimize storage."""
        joblib.dump(self.model, path, compress=3)

    @staticmethod
    def model_size(path):
        """Quantifies the storage footprint of the saved model in Megabytes (MB)."""
        if not os.path.exists(path): return 0
        return os.path.getsize(path) / (1024 * 1024)

    def complexity(self):
        """Estimates model structural complexity based on total learnable parameters."""
        param_count = sum(p.numel() for p in self.model.parameters())
        if param_count < 5000:
            return "low"
        elif param_count < 20000:
            return "medium"
        else:
            return "high"

    def get_hyperparameters(self):
        """Returns the structural and optimization parameters for experiment tracking."""
        return self.params_cfg