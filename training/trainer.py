import time
import psutil
import os
import gc
import torch
import tracemalloc
from config.params import PATH
from metrics.metrics import rmse, mae
from utils.time_tools import format_time

# training process
def train_and_evaluate(model_pro, X_train, X_val, X_test, y_train, y_val, y_test, cfg):
    # Metadata extraction
    exp_name = cfg["exp_name"]
    model_type = cfg["model"]

    # Retrieve baseline memory consumption of the current process
    process = psutil.Process(os.getpid())
    base_mem_mb = process.memory_info().rss / (1024 * 1024)

    # Initialize high-precision memory tracking
    tracemalloc.start()
    gc.collect()  # Forced garbage collection to ensure a clean baseline

    # Execution phase: Model Training
    train_start_time = time.time()

    # Route training logic based on model characteristics
    if model_type in ["linear", "rf"]:
        # Non-iterative models: Direct fitting without validation-based early stopping
        model_pro.train(X_train, y_train)
        eff_epochs = 1
    elif model_type == "mlp":
        # Iterative models: Utilize validation set for convergence monitoring (Early Stopping)
        eff_epochs = model_pro.train(
            X_train, y_train, X_val, y_val,
            epochs=cfg.get("epochs", 50),
            batch_size=cfg.get("batch_size", 32)
        )
    train_end_time = time.time()
    train_time = train_end_time - train_start_time

    # Capture peak memory utilization during the execution window
    # _, peak_delta_bytes: Returns the maximum memory increment (in Bytes) since start
    _, peak_delta_bytes = tracemalloc.get_traced_memory()
    peak_delta_mb = peak_delta_bytes / (1024 * 1024)

    # Total Peak RAM = Static Baseline + Runtime Dynamic Peak
    total_peak_ram = base_mem_mb + peak_delta_mb
    tracemalloc.stop()

    # Evaluation phase: Inference Latency Profiling
    predict_start_time = time.time()
    y_predict = model_pro.predict(X_test)
    predict_end_time = time.time()
    inference_time = predict_end_time - predict_start_time

    # Model Serialization (Storage Footprint Assessment)
    # Use .pth for PyTorch/MLP and .pkl for Scikit-learn based models
    ext = ".pth" if model_type == "mlp" else ".pkl"
    model_dir = PATH["exp_model_folder"]
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{exp_name}_model{ext}")
    model_pro.save(model_path)

    # Performance metric calculation
    test_rmse = rmse(y_test, y_predict)

    # Resource Efficiency Scoring
    # Applying stabilization offsets to prevent numerical instability in high-performance scenarios
    BASE_RAM_GB = 0.05
    BASE_TIME = 0.001

    safe_rmse = max(test_rmse, 1.0)  # Lower-bound error to prevent score explosion
    ram_gb = (total_peak_ram / 1024) + BASE_RAM_GB
    safe_time = inference_time + BASE_TIME

    # Efficiency Formula: 100 / (Error * Latency * Memory)
    # Higher scores represent superior Pareto-optimality between accuracy and resources.
    resource_eff = round(100 / (safe_rmse * safe_time * ram_gb), 6)

    # Consolidate results for telemetry and logging
    result = {
        # === Computational Performance ===
        "train_start_time": format_time(train_start_time),
        "train_end_time": format_time(train_end_time),
        "train_time": round(train_time, 6),
        "effective_epochs": eff_epochs,
        "peak_ram": round(total_peak_ram, 4),
        "avg_ram": round((base_mem_mb + total_peak_ram) / 2, 4),
        "model_size": round(model_pro.model_size(model_path), 8),

        # === Predictive Precision & Efficiency ===
        "rmse": round(test_rmse, 4),
        "mae": round(mae(y_test, y_predict), 4),
        "inference_time": round(inference_time, 4),
        "inference_batch_size": len(X_test),
        "resource_efficiency": resource_eff,
    }

    # Memory deallocation and resource release
    del model_pro
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    return result