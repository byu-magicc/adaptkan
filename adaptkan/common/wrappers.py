# ------------------------------------------------------------------------
# Advancing Out-of-Distribution Detection via Local Neuroplasticity
# Copyright (c) 2024 Alessandro Canevaro. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from OpenOOD (https://github.com/Jingkang50/OpenOOD)
# Copyright (c) 2021 Jingkang Yang. All Rights Reserved.
# ------------------------------------------------------------------------

from typing import Callable, List, Type, Any, Dict
import numpy as np
from openood.evaluators.metrics import compute_all_metrics
from openood.evaluation_api.evaluator import Evaluator
from openood.postprocessors import BasePostprocessor
from torch.nn.modules import Module
import optuna
import pdb
import os


def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
seed_everything(42)


class EvaluatorWrapper(Evaluator):

    def __init__(self, 
                 net: Module, 
                 id_name: str, 
                 data_root: str = './data', 
                 config_root: str = './configs', 
                 preprocessor: Callable[..., Any] = None, 
                 postprocessor_name: str = None, 
                 postprocessor: type[BasePostprocessor] = None, 
                 batch_size: int = 200, 
                 shuffle: bool = False, 
                 num_workers: int = 4) -> None:
        
        self.postprocessor_name = postprocessor_name
        super().__init__(net, 
                         id_name,
                         data_root, 
                         config_root, 
                         preprocessor, 
                         postprocessor_name, 
                         postprocessor, 
                         batch_size, 
                         shuffle, 
                         num_workers)

    def hyperparam_search(self):
        """
        Perform hyperparameter search using Optuna.
        
        Args:
            n_trials: Number of trials for optimization
            timeout: Stop study after this many seconds
            param_config: Optional dictionary specifying parameter types and ranges
                         Format: {'param_name': {'type': 'categorical'|'float'|'int', 
                                                 'choices': [...] or 'range': [min, max],
                                                 'log': True/False (for float only)}}
        """
        print('Starting automatic parameter search with Optuna...')
        
        # Build parameter configuration from postprocessor args if not provided
        
        def objective(trial):
            # Sample hyperparameters based on configuration
            hyperparam = []

            hyperparam_names = list(self.postprocessor.args_dict.keys())
            
            for name in hyperparam_names:
                if name in ["n_trials", "timeout"]:
                    # Skip these parameters as they are not hyperparameters
                    continue

                param_values = self.postprocessor.args_dict[name]

                if name == 'lr':
                    # Special case for learning rate: suggest float with log scale
                    min_lr, max_lr = param_values
                    value = trial.suggest_float(name, float(min_lr), float(max_lr), log=True)
                elif name == 'msp_weight':
                    min_weight, max_weight = param_values
                    value = trial.suggest_float(name, float(min_weight), float(max_weight), log=False)
                else:
                    # General case for other parameters
                    value = trial.suggest_categorical(name, param_values)

                hyperparam.append(value)
            
            # Set hyperparameters
            self.postprocessor.set_hyperparam(hyperparam)
            
            seed_everything(42)
            self.postprocessor.setup_flag = False
            self.postprocessor.setup(
                self.net, 
                self.dataloader_dict['id'], 
                self.dataloader_dict['ood']
            )
            
            # Perform inference
            id_pred, id_conf, id_gt = self.postprocessor.inference(
                self.net, self.dataloader_dict['id']['val']
            )
            ood_pred, ood_conf, ood_gt = self.postprocessor.inference(
                self.net, self.dataloader_dict['ood']['val']
            )
            
            # Compute metrics
            ood_gt = -1 * np.ones_like(ood_gt)  # hard set to -1 as ood
            pred = np.concatenate([id_pred, ood_pred])
            conf = np.concatenate([id_conf, ood_conf])
            label = np.concatenate([id_gt, ood_gt])
            ood_metrics = compute_all_metrics(conf, label, pred)
            auroc = ood_metrics[1]
            
            print(f'Trial {trial.number}: Hyperparam: {hyperparam}, AUROC: {auroc:.4f}')
            
            return auroc  # Optuna maximizes by default
        
        db_root = "results/ood"
        if not os.path.exists(db_root):
            os.makedirs(db_root)
        
        if hasattr(self.postprocessor, 'dataset_name'):
            storage_url = f'sqlite:///{db_root}/{self.postprocessor_name}_{self.postprocessor.dataset_name}_optuna.db'
            study_name = f'{self.postprocessor_name}_{self.postprocessor.dataset_name}_hyperparam_search'
        else:
            storage_url = f'sqlite:///{db_root}/{self.postprocessor_name}_optuna.db'
            study_name = f'{self.postprocessor_name}_hyperparam_search'
        
        # Create study and optimize
        study = optuna.create_study(
            storage=storage_url,
            study_name=study_name,
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42),  # For reproducibility
            load_if_exists=True
        )
        
        study.optimize(
            objective, 
            n_trials=self.postprocessor.args_dict.get('n_trials', 100),
            timeout=self.postprocessor.args_dict.get('timeout', None)  # Use timeout if specified
        )
        
        # Set best hyperparameters
        best_trial = study.best_trial
        best_hyperparam = []
        hyperparam_names = list(self.postprocessor.args_dict.keys())
        
        for name in hyperparam_names:
            if name in best_trial.params:
                best_hyperparam.append(best_trial.params[name])
            else:
                # Fixed parameter (not in sweep)
                param_values = self.postprocessor.args_dict[name]
                if isinstance(param_values, list):
                    best_hyperparam.append(param_values[0])  # Use first value as default
                else:
                    best_hyperparam.append(param_values)
        
        self.postprocessor.set_hyperparam(best_hyperparam)
        
        print(f'Best trial: {best_trial.number}')
        print(f'Best AUROC: {best_trial.value:.4f}')
        print(f'Best hyperparameters: {best_hyperparam}')
        
        self.postprocessor.hyperparam_search_done = True