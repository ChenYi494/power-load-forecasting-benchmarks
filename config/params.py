import os

# Project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Centralized path configuration
PATH = {
    # 目錄
    "dataset_folder": "data",
    "exp_config_folder": "experiments/configs",
    "exp_result_folder": "experiments/results",
    "exp_model_folder": "experiments/results/models",
    "exp_log_folder": "experiments/results/train_logs",
    # 檔案
    "source_data": "source_data.parquet",
    "linear_cfg": "linear.json",
    "mlp_cfg": "mlp.json",
    "rf_cfg": "rf.json",
    "summary_results": "summary_results.csv"
}

# Base configuration shared across all model types
COMMON_BASE = {
    "dataset_path": os.path.join(PATH['dataset_folder'], PATH['source_data']),
    "target_col": "load",
    "train_ratio": 0.8,
    "shuffle": False,
    "scaling": "standard",
    "missing_strategy": "drop",
    "num_threads": 1,
}

# Experimental parameters for Multi-Layer Perceptron (MLP)
LINEAR_PARAMS = {
    # Static configuration
    "base_cfg": {
        **COMMON_BASE,
        "model": "linear",
        "fit_intercept": True,
    },
    # Hyperparameter search space
    "dataset_sizes": [1000, 10000, 100000],
    "random_seeds": [42, 123, 999],
    "regulars": ["none", "L2"],
    "alphas": [0.01, 1.0, 10.0]
}

# Experimental parameters for Random Forest (RF)
MLP_PARAMS = {
    # Static configuration
    "base_cfg": {
        **COMMON_BASE,
        "model": "mlp",
        "learning_rate": 1e-03,
        "epochs": 200,
        "optimizer": "Adam",
        "early_stopping": "on",
        "num_threads": 1,
    },
    # Architecture and capacity scaling parameters
    "dataset_sizes": [1000, 10000, 100000],
    "random_seeds": [42, 123, 999],
    "hidden_layers": [[64, 32], [128, 64], [256, 128, 64]],
    "dropouts": [0.0, 0.3],
    "batch_sizes": [32, 256],
}

# Random Forest 模型參數
RF_PARAMS = {
    # Static configuration
    "base_cfg": {
        **COMMON_BASE,
        "model": "rf",
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "bootstrap": True,
        "num_threads": 1
    },
    # Complexity and inference jitter parameters
    "dataset_sizes": [1000, 10000, 100000],
    "random_seeds": [42, 123, 999],
    "n_estimators": [50, 100],
    "max_depth": [10, 50, None]
}

