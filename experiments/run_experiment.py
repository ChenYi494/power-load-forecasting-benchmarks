import json
import random
import numpy as np
import pandas as pd
from data.dataset import load_dataset
from models.linear_regression import LinearModel
from models.fully_connected_MLP import MLPModel
from models.random_forest import RFModel
from training.trainer import train_and_evaluate
from utils.get_hardware_profile import get_hardware_profile

# experiment process
def run_experiment(cfg):
    # Step 0: Global Seed Initialization
    # Ensures experimental reproducibility across stochastic operations
    seed = cfg["random_seed"]
    random.seed(seed)
    np.random.seed(seed)

    # Step 1: Data Acquisition
    # Temporal splitting is handled within load_dataset to prevent data leakage
    X_train, X_val, X_test, y_train, y_val, y_test, df = load_dataset(
        cfg["dataset_path"], cfg["target_col"], 1- cfg["train_ratio"], cfg["dataset_size"], cfg["shuffle"]
    )

    # Model-Specific Data Routing
    if cfg["model"] in ["rf", "linear"]:
        # Merge validation set back into training for models that do not require
        # real-time monitoring (Early Stopping), maximizing sample utilization.
        X_train = np.r_[X_train, X_val]
        y_train = np.r_[y_train, y_val]
        print(f"[{cfg['model'].upper()}] Validation set merged into training. Total samples: {len(X_train)}")

    # Step 2: Model Factory Initialization
    model_pro = get_model(cfg, X_train)

    # Step 3: Training and Resource Profiling
    results = train_and_evaluate(
        model_pro, X_train, X_val, X_test, y_train, y_val, y_test, cfg
    )

    # Final Quantitative Summary Construction
    actual_val_size = 0 if cfg["model"] in ["rf", "linear"] else len(X_val)
    summary = {
        # === Metadata & Environment ===
        "exp_name": cfg["exp_name"],
        "hardware_profile": json.dumps(get_hardware_profile()),

        # === Experimental Design ===
        "model": cfg["model"],
        "dataset_size": cfg["dataset_size"],
        "train_size": len(X_train),
        "test_size": len(X_test),
        "val_size": actual_val_size,
        "feature_num": X_train.shape[1],
        "complexity": model_pro.complexity(),
        "hyperparameters": model_pro.get_hyperparameters(),
        "seed": seed,
        "data_time_span": [str(pd.to_datetime(df['timestamp'].min())), str(pd.to_datetime(df['timestamp'].max()))],

        # === Computational Benchmarking ===
        "train_time": results["train_time"],
        "peak_ram": results["peak_ram"],
        "avg_ram": results["avg_ram"],
        "model_size": results["model_size"],
        "effective_epochs": results.get("effective_epochs", 1),

        ## === Performance & Efficiency Metrics ===
        "rmse": results["rmse"],
        "mae": results["mae"],
        "inference_time": results["inference_time"],
        "inference_batch_size": results["inference_batch_size"],
        "resource_efficiency": results["resource_efficiency"]
    }

    return summary


def get_model(cfg, X_train):
    if cfg["model"] == "linear":
        return LinearModel(
            reg_type=cfg.get("regularization", "none"),
            alpha=cfg.get("alpha", 1.0)
        )
    elif cfg["model"] == "mlp":
        input_dim = X_train.shape[1]  # Automated feature dimension mapping
        return MLPModel(
        input_dim=input_dim,
        hidden_layers=cfg["hidden_layers"],
        dropout=cfg["dropout"],
        lr=cfg["learning_rate"]
    )
    elif cfg["model"] == "rf":
        return RFModel(
            n_estimators=cfg["n_estimators"],
            max_depth=cfg["max_depth"],
            min_samples_leaf=cfg.get("min_samples_leaf", 1),
            max_features=cfg.get("max_features", "sqrt"),
            random_seed=cfg["random_seed"]
        )
    else:
        return "unknown"