from copy import deepcopy
import glob
import os
import json

def create_configs(default_config, changes):
    new_configs = []
    for change in changes:
        config = deepcopy(default_config)
        for key, value in change.items():
            if key in config:
                config[key] = value
            else:
                raise ValueError(f"Invalid config key: {key}")
        new_configs.append(config)
    return new_configs

def load_existing_results(results_dir):
    """Load all existing result files and extract their configs"""
    existing_configs = []
    
    # Find all JSON files in the directory
    json_files = glob.glob(os.path.join(results_dir, "*.json"))
    
    for file_path in json_files:
        if "cherry_picked_results" in file_path or "best_results" in file_path:
            continue
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Extract the config parameters we care about
            config = {
                "depth": data.get("depth"),
                "width": data.get("width"), 
                "activation": data.get("activation"),
                "learning_rate": data.get("learning_rate"),
                "prune_rounds": data.get("prune_rounds"),
                "lamb": data.get("lamb"),
                "steps": data.get("steps"),
                "grid": data.get("grid"),
                "adapt_first": data.get("adapt_first", None),
                "optimizer": data["optimizer"],
                "stretch_mode": data.get("stretch_mode", None),
                "prune_mode": data.get("prune_mode", None),
                "ema_alpha": data.get("ema_alpha", None),
                "prune_patience": data.get("prune_patience", None),
                "activation_strategy": data.get("activation_strategy", "linear"),
                "activation_noise": data.get("activation_noise", 0.1),
                "exact_refit": data.get("exact_refit", True),
                "seed": data.get("seed", None),
                "manual_adapt_every": data.get("manual_adapt_every", None)
            }
            existing_configs.append(config)
            
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error reading {file_path}: {e}")
            continue

        if config["width"] is None:
            print(f"Warning: 'width' is None in {file_path}")
    
    return existing_configs

def delete_files_by_signature(results_dir, target_signatures):
    """
    Delete all JSON files matching the given config signatures
    
    Args:
        results_dir: Directory containing the result files
        target_signatures: List of config dictionaries to match against
                          Each dict should contain the keys you want to match on
    
    Returns:
        List of deleted file paths
    """
    deleted_files = []
    json_files = glob.glob(os.path.join(results_dir, "*.json"))
    
    for file_path in json_files:
        # Skip special files
        if "cherry_picked_results" in file_path or "best_results" in file_path:
            continue
            
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Extract config from file
            file_config = {
                "depth": data.get("depth"),
                "width": data.get("width"),
                "activation": data.get("activation"),
                "learning_rate": data.get("learning_rate"),
                "prune_rounds": data.get("prune_rounds"),
                "lamb": data.get("lamb"),
                "steps": data.get("steps"),
                "grid": data.get("grid"),
                "adapt_first": data.get("adapt_first", None)
            }
            
            # Check if this config matches any target signature
            if get_config_signature(file_config) in target_signatures:
                os.remove(file_path)
                deleted_files.append(file_path)
                print(f"Deleted: {file_path}")
                    
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error reading {file_path}: {e}")
            continue
    
    return deleted_files

def get_config_signature(config, model_type):
    """Create a hashable signature for a config"""
    # Convert learning_rate list to tuple for hashing
    lr = tuple(config["learning_rate"]) if isinstance(config["learning_rate"], list) else config["learning_rate"]
    
    if model_type in ["AdaptKAN", "KAN"]:
        steps = tuple(config["steps"]) if isinstance(config["steps"], list) else config["steps"]
        grid = tuple(config["grid"]) if isinstance(config["grid"], list) else config["grid"]
        config["width"][0] = None
        if isinstance(config["width"], list):
            if isinstance(config["width"][1], list):
                width = tuple(w[0] if w is not None else None for w in config["width"])
            else:
                width = tuple(config["width"])
        else:
            width = config["width"]
        return (width,
                lr,
                config["prune_rounds"],
                config["lamb"], 
                steps,
                grid, 
                config["optimizer"],
                config["stretch_mode"],
                config["prune_mode"],
                config["ema_alpha"],
                config["prune_patience"],
                config["activation_strategy"],
                config["activation_noise"],
                config["exact_refit"],
                config["manual_adapt_every"],
                config["seed"])
    else:
        return (config["depth"], config["width"], config["activation"], lr, config["optimizer"])