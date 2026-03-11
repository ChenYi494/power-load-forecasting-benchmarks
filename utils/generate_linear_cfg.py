from config.params import BASE_DIR, PATH, LINEAR_PARAMS
from itertools import product
import json
import os

# Define output path and ensure the target directory exists
output_path = os.path.join(BASE_DIR, PATH["exp_config_folder"], PATH["linear_cfg"])
os.makedirs(os.path.dirname(output_path), exist_ok=True)

def generate_linear_cfg(path):
    """
    Generates a list of experimental configurations for Linear models by
    calculating the Cartesian product of defined hyperparameter dimensions.
    """

    cfg_list = []

    # Extract experimental dimensions from centralized parameters
    alphas = LINEAR_PARAMS["alphas"]
    dataset_sizes = LINEAR_PARAMS["dataset_sizes"]
    random_seeds = LINEAR_PARAMS["random_seeds"]

    # Define regularization schemes:
    # 'none' represents OLS (no alpha needed), 'L2' represents Ridge (mapped to multiple alphas)
    reg_schemes = [{"reg": "none", "alpha": None}]

    for a in alphas:
        reg_schemes.append({"reg": "L2", "alpha": a})

    combinations = product(dataset_sizes, random_seeds, reg_schemes)

    for i, (ds, seed, scheme) in enumerate(combinations, 1):
        reg_type = scheme["reg"]
        alpha_val = scheme["alpha"]

        alpha_str = f"a{alpha_val}" if alpha_val is not None else "na"
        exp_name = f"linear_ds{ds}_s{seed}_{reg_type}_{alpha_str}_{i:03d}"

        cfg = {
            "exp_name": exp_name,
            **LINEAR_PARAMS["base_cfg"],
            "dataset_size": ds,
            "random_seed": seed,
            "regularization": reg_type,
            "alpha": alpha_val
        }
        cfg_list.append(cfg)

    with open(path, "w") as f:
        json.dump(cfg_list, f, indent=2, ensure_ascii=False)

    print(f"Linear configuration generation complete. Total trials: {len(cfg_list)}")
    return cfg_list


generate_linear_cfg(output_path)