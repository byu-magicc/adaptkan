from adaptkan.common.process_feynman import get_feynman_dataset, convert_dimensionless
from kan.utils import create_dataset
import numpy as np
import torch
import json
import equinox as eqx
from adaptkan.jax.model import AdaptKANJax
# from archive.AdaptKAN_jax import mse_loss, AdaptKANJax # , AdaptKANJax
from adaptkan.jax.losses import mse_loss, mse_loss_with_reg, rmse_loss
from adaptkan.jax.fit import fit, evaluate
import jax.numpy as jnp
import uuid
from adaptkan.jax.prospective_configuration import pc_loss_with_mse, pc_test_loss_with_mse
import jax
from copy import deepcopy
from adaptkan.common.sweep_util import create_configs
from itertools import product
import os
from adaptkan.common.sweep_util import load_existing_results, get_config_signature, delete_files_by_signature
import sys
from kan import KAN

jax.config.update("jax_enable_x64", True)

# # Need to make sure the tensor that's getting converted isn't on the GPU
os.environ["JAX_PLATFORM_NAME"] = "cpu"
jax.config.update("jax_platform_name", "cpu")

# jax.config.update('jax_disable_jit', True)

print(f"sys.argv: {sys.argv}")
print(f"SLURM_ARRAY_TASK_ID: {os.environ.get('SLURM_ARRAY_TASK_ID')}")
print(f"len(sys.argv): {len(sys.argv)}")

import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# Default configuration files for each model type
default_adaptkan_config = {
    "steps": [
        2000,
        2000,
        2000,
        2000,
        2000,
    ],
    "optimizer": "Adam",
    "learning_rate": [
        0.01,
        0.01,
        0.001,
        0.001,
        0.0001,
    ],
    "dataset_size": 1000,
    "grid": [
        3,
        5,
        10,
        20,
        50,
    ],
    "prune_patience": 1,
    "k": 3,
    "stretch_mode": "half_max",
    "prune_mode": "default",
    "stretch_threshold": None,
    "ema_alpha": 0.001,
    "exact_refit": True,
    "seed": 2293716,
    "rounding_epsilon": 0.0,
    "regularization_lambda": 0.0,
    "adapt_first": False,
    "manual_adapt_every": None,
    "prune_rounds": 2,
    "lamb": 0.001,
    "width": [None, 5, 1],
    "formula_type": "dimensionless",
    "activation_strategy": "kan",
    "activation_noise": 0.5,
    "exact_refit": True,
}

default_mlp_config = {
    "steps": [
        200, # Covers time pruning
        200,
        200,
        200,
        200,
        200,
        200,
    ],
    "optimizer": "LBFGS",
    "learning_rate": [
        0.01,
        0.01,
        0.01,
        0.001,
        0.0001,
        0.00001,
        0.000001,
    ],
    "dataset_size": 1000,
    "depth": 6,
    "width": 5,
    "activation": "tanh",
    "seed": 2293716,
    "regularization_lambda": 0.0,
    "formula_type": "dimensionless"
}

default_kan_config = {
    "steps": [
        200,
        200,
        200,
        200,
        200,
    ],
    "optimizer": "LBFGS",
    "learning_rate": [
        0.01,
        0.01,
        0.001,
        0.001,
        0.0001,
    ],
    "dataset_size": 1000,
    "grid": [
        3,
        5,
        10,
        20,
        50,
    ],
    "k": 3,
    "seed": 2293716,
    "prune_rounds": 2,
    "lamb": 0.001,
    "lamb_l1": 1.0,
    "lamb_entropy": 2.0,
    "width": [None, 5, 5, 5, 1],
    "formula_type": "dimensionless",
}

def run_config(config, config_idx, save_root="/home/jmm1995/adaptkan/results/feynman/AdaptKAN_w_pruning_64/", model_type="AdaptKAN"):
    print(f"Running config {config_idx+1}")

    loss_map = dict()
    sizes_map = dict()
    params_map = dict()
    grid_map = dict()

    print("Config:", config)

    seed = config["seed"]

    print("Evaluating seed", seed)

    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

    for name in formulas:

        if config["formula_type"] is not None:
            name = "_".join([name, config["formula_type"]])

        symbol, expr, f, ranges = get_feynman_dataset(name)
        dataset = create_dataset(f, n_var=len(symbol), device=device, train_num=config["dataset_size"], ranges=ranges, seed=seed)
        lr = config["learning_rate"]

        X_train, y_train = dataset["train_input"], dataset["train_label"]
        X_test, y_test = dataset["test_input"], dataset["test_label"]

        # Use jax with adaptkan and the mlp
        X_train = jnp.array(X_train)
        y_train = jnp.array(y_train)
        X_test = jnp.array(X_test)
        y_test = jnp.array(y_test)

        test_loss_fn = mse_loss

        if model_type in ["AdaptKAN", "KAN"]:
            width = config["width"]
            width[0] = X_train.shape[1]
            grid_list = config["grid"] if isinstance(config["grid"], list) else [config["grid"]]
        else:
            grid_list = [None for _ in range(len(config["steps"]))]
            state = None

        if model_type == "AdaptKAN":
            model, state = eqx.nn.make_with_state(AdaptKANJax)(width=width,
                                                    num_grid_intervals=config["grid"][0], 
                                                    prune_patience=config["prune_patience"],
                                                    k=config["k"],
                                                    seed=seed,
                                                    stretch_mode=config["stretch_mode"],
                                                    prune_mode=config["prune_mode"],
                                                    stretch_threshold=config["stretch_threshold"],
                                                    ema_alpha=config["ema_alpha"],
                                                    rounding_precision_eps=config["rounding_epsilon"],
                                                    exact_refit=config["exact_refit"],
                                                    activation_strategy=config["activation_strategy"],
                                                    activation_noise=config["activation_noise"])
        elif model_type == "KAN":
            model = KAN(width=width, grid=grid_list[0], seed=seed, device=device).to(device)
        else:
            key = jax.random.key(seed)
            if config["activation"] == "relu":
                activation = jax.nn.relu
            elif config["activation"] == "tanh":
                activation = jax.nn.tanh
            elif config["activation"] == "silu":
                activation = jax.nn.silu

            model = eqx.nn.MLP(
                in_size=X_train.shape[1],         # Input feature size
                out_size=1,         # Output feature size
                width_size=config["width"],      # Hidden layer width
                depth=config["depth"],            # Number of hidden layers
                activation=activation,  # Activation function
                key=key             # PRNG key for initialization
            )


        all_results = {"test_loss": []}

        steps_list = config["steps"] if isinstance(config["steps"], list) else [config["steps"]]
        learning_rate_list = config["learning_rate"] if isinstance(config["learning_rate"], list) else [config["learning_rate"]]

        if model_type in ["AdaptKAN", "KAN"]:

            # Only prune for adaptkan and kan
            for prune_round in range(config["prune_rounds"]):
                print("Pruning round", prune_round+1)

                if model_type == "AdaptKAN":
                    model, state, results = fit(model,
                                                state,
                                                train_data={"X":X_train, "y":y_train},
                                                test_data={"X":X_test, "y":y_test},
                                                learning_rate=lr[0],
                                                steps=steps_list[0],
                                                opt=config["optimizer"],
                                                loss_fn=mse_loss_with_reg,
                                                test_loss_fn=test_loss_fn,
                                                lamb=config["regularization_lambda"],
                                                loss_args={
                                                    "reset_optimizer": False,
                                                    "adapt_first": config["adapt_first"],
                                                    "n_iter": config["pc_n_iter"],
                                                    "start_gamma": config["pc_start_gamma"],
                                                    "manual_adapt_every": config["manual_adapt_every"],
                                                    "lamb": config["lamb"],
                                                    "lamb_l1": 1.0,
                                                    "lamb_entropy": 2.0,
                                                })
                    model, state = model.prune(X_train, state)
                    print("Model widths after pruning", model.width)
                else:
                    try:
                        model.fit(dataset,
                                steps=50, # steps_list[0],
                                lamb=config["lamb"],
                                # lr=learning_rate_list[0],
                                opt=config["optimizer"])
                    except Exception as e:
                        print(f"Training failed, likely due to numerical issues. Skipping this configuration.")
                        print(f"Error type: {type(e).__name__}")
                        print(f"Error message: {e}")
                        break

                    try:
                        model = model.prune(edge_th=1e-2, node_th=1e-2)
                    except RuntimeError as e:
                        print("Pruning failed, possibly due to too small a model. Continuing with current model.")
                        print(e)

                    print("Model widths after pruning", model.width)

        for idx, (steps, lr, grid) in enumerate(zip(steps_list, learning_rate_list, grid_list)):
            # Refine the model if we are adaptkan or kan
            if idx >= 1 and hasattr(model, "refine"):
                if model_type == "AdaptKAN":
                    model, state = model.refine(state, new_num_grid_intervals=grid)
                else:
                    model = model.refine(grid)

            if model_type in ["AdaptKAN", "MLP"]:
                if model_type == "AdaptKAN":
                    model, state, results = fit(model,
                                                state,
                                                train_data={"X":X_train, "y":y_train},
                                                test_data={"X":X_test, "y":y_test},
                                                learning_rate=lr,
                                                steps=steps,
                                                opt=config["optimizer"],
                                                loss_fn=mse_loss,
                                                test_loss_fn=test_loss_fn,
                                                lamb=config["regularization_lambda"],
                                                loss_args={
                                                    "reset_optimizer": False,
                                                    "adapt_first": config["adapt_first"],
                                                    "n_iter": config["pc_n_iter"],
                                                    "start_gamma": config["pc_start_gamma"],
                                                    "manual_adapt_every": config["manual_adapt_every"],
                                                })
                else:
                    model, state, results = fit(model,
                                            None,
                                            train_data={"X": X_train, "y": y_train},
                                            test_data={"X": X_test, "y": y_test},
                                            learning_rate=lr,
                                            steps=steps,
                                            opt=config["optimizer"],
                                            loss_fn=mse_loss,
                                            test_loss_fn=mse_loss,
                                            lamb=config["regularization_lambda"])
            
            
                eval_results = evaluate(model, state, {"X":X_test, "y":y_test}, seed, batch_size=-1, test_loss_fn=rmse_loss)
            
                all_results["test_loss"] += [eval_results["test_loss"]]

                lowest_test_loss = eval_results["test_loss"].item()
            else:
                # Run the KAN with Pytorch
                try:
                    model.fit(dataset, steps=steps, opt=config["optimizer"]) # , lr=lr)
                    evaluation = model.evaluate(dataset)
                except:
                    print("Training failed, likely due to numerical issues. Skipping this configuration.")
                    break
                
                if evaluation["test_loss"] != evaluation["test_loss"]:
                    print(f"Test loss is NaN for {name} with config {config_idx+1}. Skipping this configuration.")
                    break

                all_results["test_loss"] += [evaluation["test_loss"]]

                # Need to rerun the evaluation in future for AdaptKAN
                lowest_test_loss = evaluation["test_loss"]


            if name not in loss_map or lowest_test_loss < loss_map[name]:
                loss_map[name] = lowest_test_loss

                if model_type in ["AdaptKAN", "KAN"]:
                    sizes_map[name] = model.width

                    # Getting the number of jax parameters is different than the pytorch version
                    if model_type == "AdaptKAN":
                        params_map[name] = sum(x.size for x in jax.tree_util.tree_leaves(model) if isinstance(x, jnp.ndarray))
                    elif model_type == "KAN":
                        params_map[name] = sum(p.numel() for p in model.parameters() if p.requires_grad)
                    
                    grid_map[name] = grid

        if len(all_results["test_loss"]) > 0:
            overall_lowest_test_loss = min(all_results["test_loss"])
        else:
            overall_lowest_test_loss = float('nan')

        print(f"Dataset {name}: {overall_lowest_test_loss}")

    config["results"] = loss_map

    if model_type in ["AdaptKAN", "KAN"]:
        config["widths"] = sizes_map
        config["grid_sizes"] = grid_map
        config["num_params"] = params_map
    else:
        # MLP
        config["num_params"] = sum(x.size for x in jax.tree_util.tree_leaves(model) if isinstance(x, jnp.ndarray))
    
    save_path = os.path.join(save_root, f"{uuid.uuid4()}.json")

    with open(save_path, "w") as f:
        json.dump(config, f, indent=2)

if __name__=='__main__':

    # Where to save the results
    if len(sys.argv) < 3:
        model_type = "MLP"
        root = ""
    else:
        model_type = str(sys.argv[2])
        root = "/home/jmm1995/adaptkan/"

    seeds = [579923283, 3934472630, 1828409026, 2987788978, 1304314947]

    if model_type == "AdaptKAN":
        save_root = os.path.join(root, "results/feynman/AdaptKAN_5_seeds_updated")

        # Hyperparameters to sweep over
        lambdas = [0.001, 0.01, 0.0001]
        widths = [[None, 5, 1], [None, 5, 5, 5, 5, 1], [None, 5, 5, 1], [None, 5, 5, 5, 1]]
        prune_rounds = [1, 0, 2]
        # schedules = [([2000, 2000, 2000, 2000, 2000], [3, 5, 10, 20, 50], [0.001, 0.001, 0.001, 0.001, 0.001])]
        schedules = [([2000, 2000, 2000, 2000, 2000], [3, 5, 10, 20, 50], [0.01, 0.005, 0.001, 0.0005, 0.0001]),
                     ([2000, 2000, 2000, 2000, 2000], [3, 5, 10, 20, 50], [0.001, 0.0005, 0.0001, 0.00005, 0.00001])]
        optimizers = ["AdamW", "Adam"]

        changes = [{"width": w, "lamb": lamb, "prune_rounds": p, "steps": s, "grid": g, "learning_rate": lr, "optimizer": opt, "seed": seed} 
            for w, lamb, p, (s, g, lr), opt, seed in product(widths, lambdas, prune_rounds, schedules, optimizers, seeds)]
        
        all_configs = create_configs(default_adaptkan_config, changes)

    elif model_type == "KAN":
        save_root = os.path.join(root, "results/feynman/KAN_5_seeds")

        # Hyperparameters to sweep over
        lambdas = [0.01, 0.001, 0.0001]
        widths = [[None, 5, 1], [None, 5, 5, 1], [None, 5, 5, 5, 1], [None, 5, 5, 5, 5, 1]]
        prune_rounds = [1, 0, 2]
        schedules = [([200, 200, 200, 200, 200], [3, 5, 10, 20, 50], [1.0, 1.0, 1.0, 1.0, 1.0])]
        optimizers = ["LBFGS"]
        

        changes = [{"width": w, "lamb": lamb, "prune_rounds": p, "steps": s, "grid": g, "learning_rate": lr, "optimizer": opt, "seed": seed} 
            for w, lamb, p, (s, g, lr), opt, seed in product(widths, lambdas, prune_rounds, schedules, optimizers, seeds)]
        
        all_configs = create_configs(default_kan_config, changes)
    else:
        save_root = os.path.join(root, "results/feynman/MLP_5_seeds")

        depths = [2, 5, 10]  # Add more depths as needed
        widths = [5, 10, 20]
        activations = ["tanh", "relu", "silu"]
        optimizers = ["Adam", "AdamW"]
        # lrs = [[0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001]]
        lrs = [[0.001, 0.0005, 0.0001, 0.00005, 0.00001, 0.000005, 0.000001]] 
               # [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001]]
        steps = [[2000, 2000, 2000, 2000, 2000, 2000, 2000]]

        changes = [{"depth": d, "activation": a, "width": w, "learning_rate": l, "steps": s, "optimizer": opt, "seed": seed} 
                   for d, a, w, l, s, opt, seed in product(depths, activations, widths, lrs, steps, optimizers, seeds)]

        all_configs = create_configs(default_mlp_config, changes)


    existing_configs = load_existing_results(save_root)
    # print([get_config_signature(config, model_type) for config in existing_configs])
    existing_signatures = {get_config_signature(config, model_type) for config in existing_configs}
    all_signatures = {get_config_signature(config, model_type) for config in all_configs}

    # If we need to delete the files by signature, do that here
    # delete_files_by_signature(save_root, existing_signatures - all_signatures)

    unrun_configs = []
    for config in all_configs:
        config_signature = get_config_signature(config, model_type)
 
        if config_signature not in existing_signatures:
            unrun_configs.append(config)

    # This will get all the unrun configs
    configs = unrun_configs
    
    print(f"Total number of configs: {len(configs)}")

    # Get the first 10 Feynman datasets
    names = [i for i in range(1, 120)]

    torch.set_default_dtype(torch.float64)
    device = torch.device("cpu") # torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    formulas = [
        "I.6.2",
        "I.6.2b",
        "I.9.18",
        "I.12.11",
        "I.13.12",
        "I.15.3x",
        "I.16.6",
        "I.18.4",
        "I.27.6",
        "I.26.2",
        "I.29.16",
        "I.30.3",
        "I.30.5",
        "I.37.4",
        "I.40.1",
        "I.44.4",
        "I.50.26",
        "II.2.42",
        "II.6.15a",
        "II.11.17",
        "II.11.27",
        "II.35.18",
        "II.36.38",
        "II.38.3",
        "III.9.52",
        "III.10.19",
        "III.17.37"
    ]

    print(sys.argv)
    if len(sys.argv) < 2:
        config_idx = 0
    else:
        config_idx = int(sys.argv[1])
    config = configs[config_idx]

    run_config(deepcopy(config), config_idx, save_root, model_type) # Either adaptkan, kan, or mlp

    import gc
    import jax
    jax.clear_caches()
    gc.collect()
        

