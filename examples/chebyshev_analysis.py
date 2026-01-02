from kan.utils import create_dataset
from adaptkan.jax.model import AdaptKANJax
from adaptkan.jax.viz import plot_model
from adaptkan.jax.fit import fit
from adaptkan.jax.data import DataLoader
from adaptkan.jax.losses import mse_loss_with_reg, mse_loss
import jax.numpy as jnp
import equinox as eqx
import jax
import jax.random as jr
import optax
import time
import matplotlib.pyplot as plt
import torch
import numpy as np
from jax import lax

# Disable JIT for easier debugging
# jax.config.update('jax_disable_jit', True)

# Generate data to train on
torch.set_default_dtype(torch.float64)
f = lambda x: torch.exp(torch.sin(torch.pi*x[:,[0]]) + x[:,[1]]**2)
dataset = create_dataset(f, n_var=2, device="cpu", train_num=1000, ranges=[-1.,1.])
dataset['train_input'].shape, dataset['train_label'].shape

X_train = jnp.array(dataset['train_input'])
y_train = jnp.array(dataset['train_label'])
X_test = jnp.array(dataset['test_input'])
y_test = jnp.array(dataset['test_label'])

model, state = eqx.nn.make_with_state(AdaptKANJax)(width=[2,1,1],
                                                   num_grid_intervals=5, # Define the number of grid intervals for the splines
                                                   prune_patience=1, # Should be set to the number of batches per epoch. Influences when the network adapts
                                                   k=5,
                                                   seed=0,
                                                   basis_type='chebyshev') # Random seed for reproducability

model, state, results = fit(model,
                                state,
                                train_data={"X": X_train, "y": y_train}, # the pre-defined MSE loss requires X and y keys
                                test_data={"X": X_test, "y": y_test},
                                learning_rate=0.1,
                                steps=1000,
                                loss_fn=mse_loss,
                                opt="Adam")

plot_model(model, state)

