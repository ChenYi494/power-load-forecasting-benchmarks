# Performance Analysis & Resource Trade-offs for Power Load Forecasting

This framework provides an end-to-end machine learning pipeline for Electricity Load Forecasting. The primary goal is to conduct a **trade-off analysis** between different model architectures (Linear, MLP, Random Forest) regarding their **predictive accuracy** and **hardware resource efficiency** (latency, memory footprint).

---

## Key Features

* **Production-grade ETL Pipeline**: Cleans and converts raw data into high-performance `.parquet` format for optimized I/O
* **Automated Feature Engineering**: Extracts temporal features (Hour, Minute, DayOfWeek) and lag variables (Lag_1, Lag_96)
* **Multi-model Experiment Grid**: Supports automated hyperparameter sweeps for Linear Regression, MLP, and Random Forest
* **Hardware Profiling**: Precisely records **Peak RAM**, **Training Time**, **Inference Latency**, and **Model Size** for every run
* **Interactive Dashboard**: Built with Streamlit to visualize and analyze performance metrics across multiple dimensions

---

## Project Structure

```bash
├── config/
│   └── params.py            # Global paths and hyperparameter search space
├── data/                    
│   ├── dataset.py           # Data loading, normalization, and splitting logic
│   └── source_data.parquet  # Pre-processed feature dataset
├── experiments/
│   ├── configs/             # Auto-generated JSON experiment configurations
│   ├── results/             # Storage for experiment logs and model weights
│   └── run_experiment.py    # Encapsulated logic for a single experiment run
├── metrics/
│   └── metrics.py           # Evaluation metric definitions (RMSE, MAE)
├── models/                  # Model definitions (LinearModel, MLPModel, RFModel)
├── training/
│   └── trainer.py           # Training supervision and resource monitoring
├── utils/                   
│   ├── generate_linear_cfg.py # Config generator for Linear Regression
│   ├── generate_mlp_cfg.py    # Config generator for MLP
│   ├── generate_drf_cfg.py    # Config generator for Random Forest
│   └── prepare_uci_data.py    # ETL: Data cleaning and feature engineering
├── visualization/           
│   ├── app.py               # Streamlit dashboard entry point
│   ├── style.css            # UI styling
│   └── summary_result.csv   # Aggregated results for visualization
└── main.py                  # Project entry: Automated end-to-end execution

```

---

## Getting Started

### 1. Environment Setup

Ensure you are using **Python 3.9+**. Install dependencies via:

```bash
pip install pandas numpy scikit-learn torch streamlit pyarrow psutil

```

### 2. Dataset Information

The raw data is sourced from the **UCI Machine Learning Repository**:

* **Dataset**: <a href="https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014" target="_blank">ElectricityLoadDiagrams20112014</a>
* **Current Status**: The pre-processed `data/source_data.parquet` is already included. You may skip the ETL and feature engineering steps
* **Manual Processing**: To re-process the raw data, download `LD2011_2014.txt` from the link above, place it in the `data/` directory, and run `prepare_uci_data.py`

### 3. Running Experiments

Execute the following commands in order to complete the automated workflow:

```bash
# [Step 1] Optional: Re-run ETL (Convert raw .txt to Parquet)
python utils/prepare_uci_data.py

# [Step 2] Generate experiment configurations (JSON) for each model
python utils/generate_linear_cfg.py
python utils/generate_mlp_cfg.py
python utils/generate_drf_cfg.py

# [Step 3] Launch main execution: Runs grid search and profiles hardware usage
python main.py

```

### 4. Visualization & Analysis

Once experiments are complete, launch the Streamlit dashboard:

```bash
# Ensure the aggregated results are in the visualization directory
cd visualization
streamlit run app.py

```

---

## Evaluation Metrics

The project evaluates models across multiple dimensions to identify the best fit for specific deployment environments:

| Category | Metric | Description |
| --- | --- | --- |
| **Prediction** | `RMSE` / `MAE` | Measures the numerical error between predicted and actual load. |
| **Compute Cost** | `Peak RAM` | Resident Set Size (RSS) Peak. Maximum physical memory allocated during training. |
| **Inference** | `Inference Time` | Average latency for a single inference pass on the test set. |
| **Storage** | `Model Size` | Disk space occupied by the serialized model file. |
| **Total Score** | `Resource Efficiency` | A custom score balancing accuracy against resource consumption. |

### Resource Efficiency Formula

Resource Efficiency = 100 / (RMSE * Inference Time * (Peak RAM / 1024))

---

## Technical Details

* **Hardware Benchmarking**: Experiments are forced to `torch.set_num_threads(1)` to ensure consistent evaluation on a single-core CPU baseline
* **Feature Engineering**: Includes `lag_1` (15-min prior) and `lag_96` (24-hour prior) to capture load seasonality
* **Data Splitting**: Utilizes Time-series Splitting to strictly prevent data leakage
* **Atomic Persistence**: Results are appended to CSV files in real-time, ensuring data integrity even if the process is interrupted