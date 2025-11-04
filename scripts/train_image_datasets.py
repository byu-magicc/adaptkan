# Code modified from: https://github.com/alessandro-canevaro/KAN-OOD

import torch
from copy import deepcopy
from openood.evaluation_api.evaluator import Evaluator
from openood.networks import ResNet18_32x32, ResNet18_224x224
from openood.evaluation_api.datasets import data_setup, get_id_ood_dataloader
from openood.datasets import get_dataloader, get_ood_dataloader
from openood.evaluation_api.preprocessor import get_default_preprocessor
import os
from glob import glob
from adaptkan.common.postprocessors import ModelPostprocessor
from openood.utils import config
from openood.networks import get_network
import numpy as np
import tqdm
from pathlib import Path

ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')

def run(run_config):

    # Arguments
    id_data = run_config['id_data']
    batch_size = run_config['batch_size']
    model = run_config["model"] # or "kan" or "mlp" or "adaptkan"
    grid_size = run_config["grid_size"]
    width = run_config["width"]
    depth = run_config["depth"]
    num_epochs = run_config["num_epochs"]
    prune_patience = run_config["prune_patience"]
    adapt_first_epoch_only = run_config["adapt_first_epoch_only"]
    reg_lambda = run_config["reg_lambda"]
    lr = run_config["lr"]
    norm = run_config["norm"] # either 0 or 1
    # TODO still need to implement ood mode
    mode = run_config["mode"] # classification or 'ood'
    postprocessor = run_config["postprocessor"] # The type of post-hoc ood method we would like to use

    # Just default this to the original dataset we were trained on
    transfer_from = run_config["transfer_from"]
    if transfer_from is None:
        transfer_from = id_data

    MODEL = {
        'cifar10': 'resnet18_32x32',
        'cifar100': 'resnet18_32x32',
        'imagenet200': 'resnet18_224x224',
    }

    # By changing this to a different network, we can do transfer learning
    epoch = '90' if transfer_from == 'imagenet200' else '100'
    backbone_checkpoint = f'./checkpoints/{transfer_from}_{MODEL[transfer_from]}_base_e{epoch}_lr0.1_default/s0/best.ckpt'

    # Set up the config files
    config_files = [
        f'./configs/datasets/{id_data}/{id_data}.yml',
        f'./configs/networks/{MODEL[transfer_from]}.yml',
        f'./configs/pipelines/{mode}.yml',
        './configs/preprocessors/base_preprocessor.yml',
        f'./configs/postprocessors/{postprocessor}.yml'
    ]

    if mode == "ood":
        config_files.append(f'./configs/datasets/{id_data}/{id_data}_ood.yml')

    cfg = config.Config(*config_files)

    # Where to save our embeddings
    save_root = os.path.join(cfg.dataset.save_embeddings_root, f'{id_data}_from_{backbone_checkpoint.split("/")[-3]}')
    os.makedirs(save_root, exist_ok=True)

    # Get the checkpoint path
    cfg.network.checkpoint = backbone_checkpoint
    cfg.network.pretrained = True
    
    # Overload the postprocessor config
    cfg.postprocessor.model = model
    cfg.postprocessor.grid_size = grid_size
    cfg.postprocessor.width = width
    cfg.postprocessor.depth = depth
    cfg.postprocessor.epochs = num_epochs
    cfg.postprocessor.prune_patience = prune_patience
    cfg.postprocessor.adapt_first_epoch_only = adapt_first_epoch_only
    cfg.postprocessor.regularization_lambda = reg_lambda
    cfg.postprocessor.lr = lr
    cfg.postprocessor.norm = norm
    cfg.postprocessor.manual_adapt_every = run_config.get("manual_adapt_every", None)

    cfg.parse_refs()

    # Get the backbone network
    net = get_network(cfg.network).cuda()

    # Get the dataloader and save the embeddings if needed
    id_loader_dict = get_dataloader(cfg)

    ood_loader_dict = None
    if mode == "ood":
        ood_loader_dict = get_ood_dataloader(cfg)

    postprocessor = ModelPostprocessor(cfg)

    # Run the classifier
    eval_results = postprocessor.setup(net, id_loader_dict, ood_loader_dict)

    if mode == "ood":
        NotImplementedError("OOD mode is not yet implemented in this script yet.")

    return {"eval_results": eval_results, "postprocessor": postprocessor, "config": cfg}

if __name__ == '__main__':

    run_config = {
        'id_data': 'cifar10',
        'batch_size': 200,
        'model': 'kan',
        'grid_size': 10,
        'width': 10,
        'depth': 2,
        'num_epochs': 20,
        'prune_patience': 10,
        'adapt_first_epoch_only': False,
        'reg_lambda': 0.0,
        'lr': 0.0027,
        'norm': 1,
        'mode': "classification",
        "postprocessor": "classifier", # Change this in the future
        'transfer_from': "cifar10"
    }

    out = run(run_config)
    print()