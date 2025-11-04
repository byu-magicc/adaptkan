import torch
from copy import deepcopy
from openood.evaluation_api.evaluator import Evaluator
from openood.networks import ResNet18_32x32, ResNet18_224x224
from openood.evaluation_api.datasets import data_setup, get_id_ood_dataloader
from openood.evaluation_api.preprocessor import get_default_preprocessor
import os
from glob import glob
from adaptkan.common.postprocessors import ModelPostprocessor
import random
import numpy as np
import argparse

from scripts.train_image_datasets import run

## OPTUNA: Import the library
import optuna

# Put this in a different folder
def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

## OPTUNA: 1. Define the objective function that Optuna will call
def objective(trial, id_data='cifar10', model='adaptkan', mode="classification", transfer_from=None):
    # Set a seed for reproducibility within each trial
    seed_everything(42)

    # --- Hyperparameters ---
    
    batch_size = 200 # This can also be a hyperparameter if you wish

    ## OPTUNA: 2. Suggest hyperparameters using the 'trial' object
    if model == 'adaptkan' or model == 'kan':
        grid_size = trial.suggest_int("grid_size", 3, 50, step=1)
        width = trial.suggest_int("width", 5, 50, step=5)
        depth = trial.suggest_int("depth", 1, 1)
        if model == "adaptkan":
            prune_patience = trial.suggest_categorical("prune_patience", [1, 10, 100])
        else:
            prune_patience = 1
        adapt_first_epoch_only = False #  trial.suggest_categorical("adapt_first_epoch_only", [True, False]) # Just set this to True by default
        manual_adapt_every = 500
    elif model == 'mlp':
        grid_size = None # Doesn't matter in this case
        width = trial.suggest_int("width", 100, 1000, step=100)
        depth = trial.suggest_int("depth", 1, 10)   
        prune_patience = None # Not applicable to MLP
        adapt_first_epoch_only = False # MLP doesn't adapt in the same way
    else:
        raise ValueError(f"Unsupported model type: {model}")
    
    # Just fix the number of epochs for now
    num_epochs = 20
    reg_lambda = trial.suggest_categorical("reg_lambda", [0.0, 0.0001, 0.001, 0.01])
    lr = trial.suggest_float("lr", 1e-6, 1e-2, log=True)
    norm = trial.suggest_categorical("norm", [0, 1])  # Either 0 or 1

    if model == 'kan':
        norm = 0
    else:
        norm = trial.suggest_categorical("norm", [0, 1])

    run_config = {
        'id_data': id_data,
        'batch_size': batch_size,
        'model': model,
        'grid_size': grid_size,
        'width': width,
        'depth': depth,
        'num_epochs': num_epochs,
        'prune_patience': prune_patience,
        'adapt_first_epoch_only': adapt_first_epoch_only,
        'reg_lambda': reg_lambda,
        'lr': lr,
        'norm': norm,
        'mode': mode,
        "postprocessor": "classifier",
        "transfer_from": transfer_from,
        "manual_adapt_every": manual_adapt_every
    }
    
    out = run(run_config)

    # Save the number of parameters as a user attribute
    trial.set_user_attr("num_model_parameters", out["eval_results"]['num_model_params'])

    return out["eval_results"]['best_val_acc']

if __name__ == '__main__':

    ## ARGPARSE: 1. Set up the argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="adaptkan", choices=["adaptkan", "kan", "mlp"],
                        help='The model type to use for the hyperparameter search.')
    parser.add_argument('--id_data', type=str, default="cifar10", choices=['cifar10', 'cifar100', 'imagenet200'],
                        help='The in-distribution dataset to use.')
    parser.add_argument('--mode', type=str, default='classification', choices=['classification', 'ood'],
                        help='The mode of evaluation: classification or OOD detection.')
    parser.add_argument('--root_dir', type=str, default=f'/home/jmoody/Projects/KAN-OOD/results',
                        help='Root directory for saving results and models.')
    parser.add_argument('--n_trials', type=int, default=100,
                        help='Number of trials to run for the evaluation.')
    args = parser.parse_args()

    # Define a unique name for this study so you can load it later.
    study_name = f"{args.id_data}_{args.model}_search"

    # FIX: Use absolute path and ensure directory exists
    import os
    db_dir = os.path.expanduser(f'~/adaptkan/results/{args.mode}')
    os.makedirs(db_dir, exist_ok=True)
    storage_url = f"sqlite:///{os.path.join(db_dir, study_name)}.db"
    
    ## OPTUNA: 2. Create a study and run the optimization
    study = optuna.create_study(direction='maximize',
                                study_name=study_name,
                                storage=storage_url,
                                load_if_exists=True)

    root = os.path.join(args.root_dir, f'{args.id_data}_resnet18_32x32_base_e100_lr0.1_default')
    
    # 3. Use a lambda function to pass the parsed arguments to the objective
    study.optimize(lambda trial: objective(trial,
                                           args.id_data,
                                           args.model,
                                           mode=args.mode,
                                           transfer_from=args.transfer_from),
                                           n_trials=args.n_trials,
                                           catch=(Exception,))

    # Print the results
    print("Study statistics: ")
    print(f"  Number of finished trials: {len(study.trials)}")

    print("Best trial:")
    best_trial = study.best_trial
    print(f"  Value (Accuracy): {best_trial.value}")

    print("  Params: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")