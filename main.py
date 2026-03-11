import torch
import json
import os
import pandas as pd
from experiments.run_experiment import run_experiment
from config.params import PATH

config_keys = ["linear_cfg", "mlp_cfg", "rf_cfg"]
summary_path = str(os.path.join(PATH["exp_result_folder"], PATH["summary_results"]))

if __name__ == "__main__":
    # Resource Lock: Enforce single-threaded execution to ensure deterministic
    # and reproducible CPU benchmarking across different model architectures
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    os.makedirs(PATH["exp_result_folder"], exist_ok=True)

    # Systematic iteration across model categories (Linear, MLP, Random Forest)
    for key in config_keys:
        configs_path = str(os.path.join(PATH["exp_config_folder"], PATH[key]))

        if not os.path.exists(configs_path):
            print(f"Warning: Configuration file not found for {key} at {configs_path}")
            continue

        with open(configs_path) as file:
            cfg_list = json.load(file)

        print(f"\n>>> Starting Category: {key} (Total experiments: {len(cfg_list)})")

        # Execute grid search: Iterate through all parameter combinations in the category
        for cfg in cfg_list:
            print(f"Running experiment: {cfg['exp_name']}")
            results = run_experiment(cfg)

            # Atomic Persistence: Write results to a global summary CSV immediately
            # after each trial to prevent data loss in case of interruptions.
            df_new = pd.DataFrame([results])
            file_exists = os.path.isfile(summary_path)
            df_new.to_csv(summary_path, mode='a', index=False, header=not file_exists)

    print("All categories done! Global summary saved at:", summary_path)