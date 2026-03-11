from config.params import BASE_DIR, PATH, RF_PARAMS
from itertools import product
import json
import os

# Define output path and ensure the target directory exists for Random Forest configuration storage
output_path = os.path.join(BASE_DIR, PATH["exp_config_folder"], PATH["rf_cfg"])
os.makedirs(os.path.dirname(output_path), exist_ok=True)

def generate_rf_cfg(path):
    """
    Generates a structured list of experimental configurations for Random Forest models.
    Systematically explores the interaction between ensemble size and tree depth.
    """

    cfg_list = []

    # Extract experimental dimensions from the centralized RF parameter registry
    dataset_sizes = RF_PARAMS["dataset_sizes"]
    random_seeds = RF_PARAMS["random_seeds"]
    n_estimators = RF_PARAMS["n_estimators"]
    max_depths = RF_PARAMS["max_depth"]

    combinations = product(dataset_sizes, random_seeds, n_estimators, max_depths)

    for i, (ds, seed, nest, depth) in enumerate(combinations, 1):
        d_str = "unlimit" if depth is None else str(depth)
        exp_name = f"rf_ds{ds}_s{seed}_n{nest}_d{d_str}_{i:03d}"

        cfg = {
            "exp_name": exp_name,
            **RF_PARAMS["base_cfg"],
            "dataset_size": ds,
            "random_seed": seed,
            "n_estimators": nest,
            "max_depth": depth
        }
        cfg_list.append(cfg)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg_list, f, indent=2)

    print(f"Random Forest configuration generation complete. Total trials: {len(cfg_list)}")
    return cfg_list


generate_rf_cfg(output_path)