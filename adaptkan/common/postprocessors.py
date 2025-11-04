# ------------------------------------------------------------------------
# Advancing Out-of-Distribution Detection via Local Neuroplasticity
# Copyright (c) 2024 Alessandro Canevaro. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from OpenOOD (https://github.com/Jingkang50/OpenOOD)
# Copyright (c) 2021 Jingkang Yang. All Rights Reserved.
# ------------------------------------------------------------------------

from copy import deepcopy
from typing import Any
from sklearn.decomposition import PCA
from tqdm import tqdm
import gc
import numpy as np
import sklearn
import torch
import torch.nn as nn
import torch.nn.functional as F

from openood.postprocessors.base_postprocessor import BasePostprocessor
from openood.postprocessors.info import num_classes_dict

import copy
from typing import Any
from copy import deepcopy

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
from tqdm import tqdm
from scipy.stats import gaussian_kde
import equinox as eqx
from adaptkan.jax.model import AdaptKANJax

from adaptkan.jax.data import DataLoader
import jax.numpy as jnp
import jax

from efficient_kan import KAN
# from kan import KAN
from adaptkan.jax.losses import cross_entropy_loss
from adaptkan.jax.fit import fit, evaluate

from adaptkan.common.image_util import get_id_embeddings, get_ood_embeddings

import os

from copy import deepcopy
import fcntl

def safe_load_npy(filename):
    with open(filename, 'rb') as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_SH)  # Shared lock for reading
        return np.load(f, allow_pickle=True)

def safe_save_npy(filename, data):
    with open(filename, 'wb') as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)  # Exclusive lock for writing
        np.save(f, data)

def get_histogram_bins(features, num_bins=100, start=0.0, stop=1.0):
    """
    Get the bin indices for each feature value.
    
    Args:
        features: (N, 960) tensor
        num_bins: number of histogram bins
        start: minimum value for binning  
        stop: maximum value for binning
    
    Returns:
        bin_indices: (N, 960) tensor of bin indices
        bin_edges: (num_bins + 1,) tensor of bin boundaries
    """
    # Create bin edges
    bin_edges = torch.linspace(start, stop, num_bins + 1).type_as(features)
    
    # Get bin indices for all features at once
    bin_indices = torch.bucketize(features, bin_edges, right=False)
    
    # Clamp to valid range [0, num_bins-1]
    bin_indices = torch.clamp(bin_indices, 0, num_bins - 1)
    
    return bin_indices, bin_edges

def count_overlaps_chunked(query_bins, stored_bins, chunk_size=1000):
    """
    Process in chunks to avoid memory issues with large tensors
    """
    num_queries = query_bins.shape[0]
    num_stored = stored_bins.shape[0]
    
    overlap_counts = torch.zeros(num_queries, num_stored, device=query_bins.device)
    
    for i in range(0, num_stored, chunk_size):
        end_idx = min(i + chunk_size, num_stored)
        stored_chunk = stored_bins[i:end_idx]  # (chunk_size, 960)
        
        # Compare against chunk
        query_expanded = query_bins.unsqueeze(1)           # (100, 1, 960)
        chunk_expanded = stored_chunk.unsqueeze(0)         # (1, chunk_size, 960)
        
        matches = (query_expanded == chunk_expanded)       # (100, chunk_size, 960)
        chunk_overlaps = matches.sum(dim=-1)              # (100, chunk_size)
        
        overlap_counts[:, i:end_idx] = chunk_overlaps
    
    return overlap_counts

class HistBasePostprocessor():
    def __init__(self, params_config=None):        
        self.pc = self.default_config()
        self.configure(params_config)

        self.setup_flag = False

        self.a = 0. if self.pc["norm"] else -1.
        self.b = 1. if self.pc["norm"] else 3.
        self.histograms = [torch.zeros(self.pc["kan_num_inputs"], self.pc["added_hist_bins"]).to(self.pc["device"]) for _ in range(self.pc["num_partitions"])]

    @classmethod
    def default_config(cls):
        """
        Default environment configuration.
        Can be overloaded in environment implementations, or by calling configure().
        Returns:
            dict: A configuration dictionary.
        """
        return {
            "num_classes": 10,
            "grid_size": 50,
            "num_partitions": 10,
            "norm": 1,
            "device": "cpu",
            "mode": 0,
            "kan_num_inputs": 512,
            "lr": 0.1,
            "epochs": 1,
            "batch_size": 512,
            "train_size": 50000,
            "pca_comp": 16,
            "hist_bins": 15,
        }

    def normalize_data(self, data, cdf_normalized_list, bin_edges_list):
        normalized_data = torch.zeros_like(data)
        num_features = data.shape[1]
        
        for feature_idx in range(num_features):
            feature_data = data[:, feature_idx].detach().cpu().numpy()
            cdf_normalized = cdf_normalized_list[feature_idx]
            bin_edges = bin_edges_list[feature_idx]
            normalized_feature = np.interp(feature_data, bin_edges[:-1], cdf_normalized)
            normalized_data[:, feature_idx] = torch.tensor(normalized_feature)
        
        return normalized_data

    
    def hist_setup(self, all_feats, all_labels):
        print("Data prep")
        torch.autograd.set_detect_anomaly(True)
        all_feats = all_feats.to(self.pc["device"])
        all_labels = all_labels.to(self.pc["device"])
        #all_labels_reduced = all_labels_reduced.to(self.pc["device"])
        print("dataset shape", all_feats.shape)

        if self.pc["norm"]:
            cdf_normalized_list = []
            bin_edges_list = []
   
            for feature_idx in range(self.pc["kan_num_inputs"]):
                feature_data = all_feats[:, feature_idx].detach().cpu().numpy()
                hist, bin_edges = np.histogram(feature_data, bins=self.pc["hist_bins"], range=(feature_data.min(), feature_data.max()), density=True)
                cdf = hist.cumsum()
                cdf_normalized = cdf / cdf[-1]  # Normalize to range [0, 1]
                
                cdf_normalized_list.append(cdf_normalized)
                bin_edges_list.append(bin_edges)

            # Save the CDF and bin edges for each feature
            safe_save_npy(f'cdf_normalized_list.npy', cdf_normalized_list)
            safe_save_npy(f'bin_edges_list.npy', bin_edges_list)
            # np.save('cdf_normalized_list.npy', cdf_normalized_list)
            # np.save('bin_edges_list.npy', bin_edges_list)
            all_feats = self.normalize_data(all_feats, cdf_normalized_list, bin_edges_list)
            print("max, min", torch.max(all_feats), torch.min(all_feats))
                
        print("Updating histograms...")

        if self.pc["mode"] == 0: #class based
            num_samples = all_feats.shape[0] // self.pc["num_classes"]
            partitioned_data = [(all_feats[i * num_samples:(i + 1) * num_samples, :], 
                               all_labels[i * num_samples:(i + 1) * num_samples]) for i in range(self.pc["num_classes"])]
        elif self.pc["mode"] == 1:# pca+kmeans
            # Step 1: Apply PCA to the dataset
            pca = PCA(n_components=self.pc["pca_comp"])
            principal_components = pca.fit_transform(all_feats.cpu())
            
            # Step 2: Perform KMeans clustering on the PCA features
            kmeans = KMeans(n_clusters=self.pc["num_partitions"])
            cluster_assignments = kmeans.fit_predict(principal_components)

            # Step 3: Partition the data based on the cluster assignments
            partitioned_data = []
            for cluster in range(self.pc["num_partitions"]):
                cluster_indices = (cluster_assignments == cluster)
                partition_feats = all_feats[cluster_indices]
                partition_labels = all_labels[cluster_indices]
                partitioned_data.append((partition_feats, partition_labels))
        else:
            raise ValueError("Invalid mode")
        
        self.normed_hists = []
        hists = []
        entropies = []
        for i, (partition_feats, partition_labels) in enumerate(partitioned_data):
            class_hist = torch.stack([torch.histc(partition_feats[:, i], bins=self.pc["grid_size"], min=self.a, max=self.b) for i in range(partition_feats.shape[1])])
            normed_class_hist = class_hist / class_hist.sum(1, keepdims=True)  # Normalize each histogram
            self.normed_hists.append(normed_class_hist)
            class_entropy = -(normed_class_hist*torch.log2(normed_class_hist + 1e-8)).sum(-1)
            hists.append(class_hist)
            entropies.append(class_entropy)

        all_hists = torch.stack([histogram for histogram in hists]).sum(0)
        self.normed_all_hists = all_hists / all_hists.sum(1, keepdims=True)
        all_entropy = -(self.normed_all_hists*torch.log2(self.normed_all_hists + 1e-8)).sum(-1)

        # Get the indices that are most important for each class
        self.partition_indices = []
        for partition in range(self.pc["num_partitions"]):
            rel_entropy = all_entropy - entropies[partition]
            self.partition_indices.append(torch.topk(rel_entropy, k=self.pc["top_k"]).indices)

    def compute_ood_score(self, data, logits=None):
        if self.pc["norm"]:
            cdf_normalized_list = safe_load_npy(f'cdf_normalized_list.npy')
            bin_edges_list = safe_load_npy(f'bin_edges_list.npy')
            data = self.normalize_data(data, cdf_normalized_list, bin_edges_list)

        data = data.to(self.pc["device"])

        # For this mode, just default to the regular histogram method
        bin_indices, bin_edges = get_histogram_bins(data, self.pc["grid_size"], self.a, self.b)
        
        batch_size, num_features = bin_indices.shape
        probs = self.normed_all_hists[torch.arange(num_features)[None, :], bin_indices]
        hist_scores = torch.log(probs + 1e-8).sum(-1)

        if self.pc["use_msp"]:
            avg_hist_scores = hist_scores / data.shape[-1]
            # If using a single model, we just return the scores directly
            softmax_logits = F.softmax(logits, dim=-1)
            msp = softmax_logits.max(-1)[0]
            scores = avg_hist_scores + self.pc["msp_weight"] * torch.log(msp + 1e-8)
        else:
            scores = hist_scores

        return scores
    

    @staticmethod
    def _recursive_update(d, u):
        """
        Recursively update dictionary `d` with values from dictionary `u`.
        """
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = KANBasePostprocessor._recursive_update(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    def configure(self, config) -> None:
        """
        Configure.
        Args:
            config (dict): Configuration parameters.
        """
        if config:
            self.pc = self._recursive_update(self.pc, config)



class HistPostprocessor(BasePostprocessor):
    def __init__(self, config, params_config=None):
        self.config = config
        self.dataset_name = self.config["dataset"]["name"]
        self.setup_flag = False

        self.args = self.config.postprocessor.postprocessor_args
        
        self.grid_size = self.args.grid_size
        self.num_partitions = self.args.num_partitions
        self.args_dict = self.config.postprocessor.postprocessor_sweep
        self.adaptkan = self.args.adaptkan if "adaptkan" in self.args else False
        self.use_hist = self.args.use_hist if "use_hist" in self.args else False
        self.use_single_model = self.args.use_single_model if "use_single_model" in self.args else False
        self.use_msp = self.args.use_msp if "use_msp" in self.args else False
        self.norm = self.args.norm if "norm" in self.args else 1
        self.epochs = self.args.epochs if "epochs" in self.args else 1
        self.lr = self.args.lr if "lr" in self.args else 0.1
        self.reg_lambda = self.args.reg_lambda if "reg_lambda" in self.args else 0.0
        self.softmax_div = self.args.softmax_div if "softmax_div" in self.args else 1.0
        self.batch_size = self.args.batch_size if "batch_size" in self.args else 512    
        self.top_k = self.args.top_k if "top_k" in self.args else 100
        self.hist_agg_mode = self.args.hist_agg_mode if "hist_agg_mode" in self.args else "min"
        self.msp_weight = self.args.msp_weight if "msp_weight" in self.args else 0.1
        self.hist_interaction_mode = self.args.hist_interaction_mode if "hist_interaction_mode" in self.args else False
        # These are the number of bins for the added histogram
        self.added_hist_bins = self.args.added_hist_bins if "added_hist_bins" in self.args else self.grid_size

        self.pc = self.default_config()        

    @classmethod
    def default_config(cls):
        """
        Default environment configuration.
        Can be overloaded in environment implementations, or by calling configure().
        Returns:
            dict: A configuration dictionary.
        """
        return {
            "num_classes": 10,
            "grid_size": 50,
            "num_partitions": 10,
            "norm": 1,
            "device": "cuda",
            "mode": 1,
            "pca_comp": 16,
            "hist_bins": 15,
            "aggregate_layers": 1,
            "train_size": 50000,
        }
        
    def _make_hists(self):
        self.pc = self.default_config()
        params_config = {"grid_size": self.grid_size,
                         "num_partitions": self.num_partitions,
                         "adaptkan": self.adaptkan,
                         "use_hist": self.use_hist,
                         "use_msp": self.use_msp,
                         "use_single_model": self.use_single_model,
                         "norm": self.norm,
                         "added_hist_bins": self.added_hist_bins,
                         "mode": get_preprocessing_info("mode", self.dataset_name), 
                         "lr": self.lr,
                         "reg_lambda": self.reg_lambda,
                         "batch_size": self.batch_size,
                         "epochs": self.epochs,
                         "softmax_div": self.softmax_div,
                         "top_k": self.top_k,
                         "hist_agg_mode": self.hist_agg_mode,
                         "hist_bins": get_preprocessing_info("hist_bins", self.dataset_name),
                         "hist_interaction_mode": self.hist_interaction_mode,
                         "msp_weight": self.msp_weight,
                         "num_classes": get_preprocessing_info("num_classes", self.dataset_name)}
        self.configure(params_config)
        # Define model
        self.pc["kan_num_inputs"] = sum([64, 64, 128, 256, 512][self.pc["aggregate_layers"]:])
        self.hist_postprocessor = HistBasePostprocessor(params_config=self.pc)

    def get_dataloader(self, id_loader_dict):
        from copy import deepcopy
        all_data = []
        all_labels = []
        with torch.no_grad():
            for batch in id_loader_dict['train']:
                data, labels = batch['data'], batch['label']
                
                all_data.append(deepcopy(data))
                all_labels.append(deepcopy(labels))

            all_data = torch.cat(all_data)
            all_labels = torch.cat(all_labels)
        # Initialize lists to store the reduced data and labels
        reduced_data = []
        reduced_labels = []
        for class_idx in range(10):
            class_indices = (all_labels == class_idx).nonzero(as_tuple=True)[0]
            if True:#class_idx < 5:
                selected_indices = class_indices[torch.randperm(len(class_indices))[:5000]]
            else:
                selected_indices = class_indices
            reduced_data.append(all_data[selected_indices])
            reduced_labels.append(all_labels[selected_indices])
        reduced_data = torch.cat(reduced_data, dim=0)
        reduced_labels = torch.cat(reduced_labels, dim=0)
        # Verify the shape of the reduced dataset
        from torch.utils.data import TensorDataset, DataLoader

        dataset = TensorDataset(reduced_data, reduced_labels)
        dataloader = DataLoader(dataset, batch_size=100, shuffle=True)
        return dataloader

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        if self.dataset_name in ["cifar10", "cifar100"]: #reduced training size experiment
            dataloader = self.get_dataloader(id_loader_dict)
            
            if not self.setup_flag:
                self._make_hists()
            
            all_feats = []
            all_labels = []
            all_preds = []

            # Check if features are already saved
            if not os.path.exists(os.path.join(self.config.dataset.save_root, self.dataset_name, "train_feature.npy")):
                os.makedirs(os.path.join(self.config.dataset.save_root, self.dataset_name), exist_ok=True)
                with torch.no_grad():
                    if self.dataset_name in ["cifar10"]:
                        batch_count = 0
                        for batch in dataloader:
                            data, labels = batch
                            data = data.cuda()
                            
                            logits, features_list = net(data, return_feature_list=True)

                            features = torch.concatenate([layer.mean(dim=(2, 3)) for layer in features_list[self.pc["aggregate_layers"]:]], dim=1)
                            
                            all_feats.append(features.cpu())
                            all_labels.append(deepcopy(labels))
                            all_preds.append(logits.argmax(1).cpu())
                            
                            batch_count += 1
                            
                    else:
                        batch_count = 0
                        for batch in id_loader_dict['train']:
                            data, labels = batch['data'].cuda(), batch['label']
                            
                            logits, features = net(data, return_feature=True)
                            logits, features_list = net(data, return_feature_list=True)

                            features = torch.concatenate([layer.mean(dim=(2, 3)) for layer in features_list[self.pc["aggregate_layers"]:]], dim=1)
                            all_feats.append(features.cpu())
                            all_labels.append(deepcopy(labels))
                            all_preds.append(logits.argmax(1).cpu())
                            
                            batch_count += 1
                
                all_feats = torch.cat(all_feats)
                all_labels = torch.cat(all_labels)
                all_preds = torch.cat(all_preds)

                np.save(os.path.join(self.config.dataset.save_root, self.dataset_name, "train_feature.npy"), all_feats.cpu().numpy())
                np.save(os.path.join(self.config.dataset.save_root, self.dataset_name, "train_labels.npy"), all_labels.cpu().numpy())
                np.save(os.path.join(self.config.dataset.save_root, self.dataset_name, "train_logits.npy"), all_preds.cpu().numpy())

            else:
                data, labels, logits, features_list, features = None, None, None, None, None
                all_feats = torch.from_numpy(np.load(os.path.join(self.config.dataset.save_root, self.dataset_name, "train_feature.npy"))).cuda()
                all_labels = torch.from_numpy(np.load(os.path.join(self.config.dataset.save_root, self.dataset_name, "train_labels.npy"))).cuda()
                all_preds = torch.from_numpy(np.load(os.path.join(self.config.dataset.save_root, self.dataset_name, "train_logits.npy"))).cuda()
            
            # sanity check on train acc
            print(all_preds.shape, all_labels.shape)
            train_acc = all_preds.eq(all_labels).float().mean()
            print(f' Train acc: {train_acc:.2%}')
            
            self.hist_postprocessor.hist_setup(all_feats, all_labels)
            self.setup_flag = True
            
            # Free GPU memory
            del data, labels, logits, features_list, features, all_feats, all_labels, all_preds
            torch.cuda.empty_cache()
            gc.collect()
        else:
            pass

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        logits, features_list = net(data, return_feature_list=True)
        features = torch.concatenate([layer.mean(dim=(2, 3)) for layer in features_list[self.pc["aggregate_layers"]:]], dim=1)

        pred = logits.argmax(1)
        
        if self.pc["use_msp"]:
            # Use maximum softmax probability
            scores = self.hist_postprocessor.compute_ood_score(features, logits=logits)
        else:
            scores = self.hist_postprocessor.compute_ood_score(features)

        conf = torch.tensor(scores).float() 
        return pred, conf
    
    @staticmethod
    def _recursive_update(d, u):
        """
        Recursively update dictionary `d` with values from dictionary `u`.
        """
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = KANPostprocessor._recursive_update(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    def configure(self, config) -> None:
        """
        Configure.
        Args:
            config (dict): Configuration parameters.
        """
        if config:
            self.pc = self._recursive_update(self.pc, config)

    def set_hyperparam(self, hyperparam: list):
        if len(hyperparam) == 3:
            self.grid_size = hyperparam[0]
            self.norm = hyperparam[1]
            self.msp_weight = hyperparam[2]
        elif len(hyperparam) == 4:
            self.grid_size = hyperparam[0]
            self.reg_lambda = hyperparam[1]
            self.lr = hyperparam[2]
            self.norm = hyperparam[3]
        elif len(hyperparam) == 5:
            self.grid_size = hyperparam[0]
            self.num_partitions = hyperparam[1]
            self.top_k = hyperparam[2]
            self.norm = hyperparam[3]
            self.hist_agg_mode = hyperparam[4]
        else:
            self.grid_size = hyperparam[0]
            self.reg_lambda = hyperparam[1]
            self.lr = hyperparam[2]
            self.norm = hyperparam[3]
            self.softmax_div = hyperparam[4]
            self.batch_size = hyperparam[5]
            self.epochs = hyperparam[6]

    def get_hyperparam(self):
        return [self.grid_size,
                self.reg_lambda,
                self.lr,
                self.norm,
                self.softmax_div,
                self.batch_size,
                self.epochs,
                self.num_partitions,
                self.top_k,
                self.hist_agg_mode,
                self.msp_weight]


def get_preprocessing_info(name, dataset):
    if name == "mode":
        return 1 if dataset == "cifar10" else 0
    if name == "hist_bins":
        return 5 if dataset == "cifar10" else 10
    if name == "num_classes":
        if dataset == "cifar10":
            return 10
        elif dataset == "cifar100": 
            return 100
        elif dataset == "imagenet200":
            return 200