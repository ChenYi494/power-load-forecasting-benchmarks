from config.params import BASE_DIR, PATH, MLP_PARAMS
from itertools import product
import json
import os

# Define output path and ensure the target directory exists for MLP configuration storage
output_path = os.path.join(BASE_DIR, PATH["exp_config_folder"], PATH["mlp_cfg"])
os.makedirs(os.path.dirname(output_path), exist_ok=True)

def generate_mlp_cfg(path):
    """
    Generates a comprehensive list of experimental configurations for MLP models.
    Maps the multidimensional hyperparameter space into a structured JSON registry.
    """

    cfg_list = []

    # Extract experimental dimensions from the centralized MLP parameter registry
    dataset_sizes = MLP_PARAMS["dataset_sizes"]
    random_seeds = MLP_PARAMS["random_seeds"]
    hidden_layers = MLP_PARAMS["hidden_layers"]
    dropouts = MLP_PARAMS["dropouts"]
    batch_sizes = MLP_PARAMS["batch_sizes"]

    combinations = product(dataset_sizes, random_seeds, hidden_layers, dropouts, batch_sizes)

    for i, (ds, seed, layers, drop, batch) in enumerate(combinations, 1):
        layer_str = "x".join(map(str, layers))
        exp_name = f"mlp_ds{ds}_s{seed}_{layer_str}_d{drop}_b{batch}_{i:03d}"

        cfg = {
            "exp_name": exp_name,
            **MLP_PARAMS["base_cfg"],
            "dataset_size": ds,
            "random_seed": seed,
            "hidden_layers": layers,
            "dropout": drop,
            "batch_size": batch
        }
        cfg_list.append(cfg)

    with open(path, "w") as f:
        json.dump(cfg_list, f, indent=2)

    print(f"MLP configuration generation complete. Total trials: {len(cfg_list)}")
    return cfg_list

generate_mlp_cfg(output_path)
