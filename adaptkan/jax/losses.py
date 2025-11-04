import jax
import jax.numpy as jnp
import optax
import equinox as eqx
from jax import vmap
import pdb
from jax.scipy.special import logsumexp
from adaptkan.jax.utils import spline_interpolate_jax

@eqx.filter_jit
def mse_loss(model, state, batch, args):
    xs = batch["X"]
    ys = batch["y"]

    if hasattr(model, 'adapt'):
        pred_ys, state = model(xs, state, update=args["update"])
    # Models without adapt() method → stateless path  
    else:
        pred_ys = vmap(model)(xs)

    # Compute loss
    loss = jnp.mean((pred_ys.squeeze() - ys.squeeze()) ** 2) # + sum(jnp.sum(jnp.square(param)) for param in jax.tree_util.tree_leaves(model)) * 0.00001
    # Return loss and a dictionary of metrics here
    return loss, ({"loss": loss, "batch_size": xs.shape[0]}, state) # Just show the rmse loss

@eqx.filter_jit
def rmse_loss(model, state, batch, args):
    xs = batch["X"]
    ys = batch["y"]

    if hasattr(model, 'adapt'):
        pred_ys, state = model(xs, state, update=args["update"])
    # Models without adapt() method → stateless path  
    else:
        pred_ys = vmap(model)(xs)

    # Compute loss
    loss = jnp.sqrt(jnp.mean((pred_ys.squeeze() - ys.squeeze()) ** 2)) # + sum(jnp.sum(jnp.square(param)) for param in jax.tree_util.tree_leaves(model)) * 0.00001
    # Return loss and a dictionary of metrics here
    return loss, ({"loss": loss, "batch_size": xs.shape[0]}, state) # Just show the rmse loss

@eqx.filter_jit
def cross_entropy_loss(model, state, batch, args):
    xs = batch["X"]
    ys = batch["y"]

    if hasattr(model, 'adapt'):
        pred_ys, state = model(xs, state, update=args["update"])
    # Models without adapt() method → stateless path  
    else:
        pred_ys = vmap(model)(xs)

    # Ensure labels are one-hot encoded
    labels_onehot = jax.nn.one_hot(ys, num_classes=pred_ys.shape[-1])
    # Compute the categorical cross-entropy loss
    loss = jnp.mean(optax.softmax_cross_entropy(pred_ys, labels_onehot))
    # Return the mean loss over all instances
    acc = (pred_ys.argmax(-1) == ys).sum() / len(ys)

    return loss, ({"loss": loss, "acc": acc, "batch_size": xs.shape[0]}, state)

@eqx.filter_jit
def binary_cross_entropy_loss(model, state, batch, args):
    """
    Dedicated binary cross-entropy loss function.
    Assumes model outputs single value per sample (sigmoid activation).
    """
    xs = batch["X"]
    ys = batch["y"].astype(jnp.int32).squeeze()  # Ensure integer labels
    
    # Handle adaptive vs stateless models
    if hasattr(model, 'adapt'):
        pred_logits, state = model(xs, state, update=args["update"])
    else:
        pred_logits = vmap(model)(xs)
    
    # Ensure single output per sample
    pred_logits = pred_logits.squeeze()
    
    # Binary cross-entropy loss
    loss = jnp.mean(optax.sigmoid_binary_cross_entropy(pred_logits, ys))
    
    # Binary predictions (threshold at 0 for logits)
    predictions = (pred_logits > 0.0).astype(jnp.int32)
    accuracy = jnp.mean(predictions == ys)
    
    metrics = {
        "loss": loss,
        "acc": accuracy,
        "batch_size": xs.shape[0]
    }
    
    return loss, (metrics, state)

@eqx.filter_jit
def mse_loss_with_reg(model, state, batch, args):
    xs = batch["X"]
    ys = batch["y"]

    # Forward pass to get predictions
    out = model.call_with_details(xs, state, update=args["update"])
    layer_act_norms = out["layer_act_norms"]
    pred_ys = out["x"]
    state = out["state"]

    # Add the l1 and entropy losses
    l1_loss = 0.
    entropy_loss = 0.
    for act_norm in layer_act_norms:
        act_norm_sum = act_norm.sum()
        l1_loss += act_norm_sum
        entropy_loss -= ((act_norm/act_norm_sum) * jnp.log(act_norm/act_norm_sum)).sum()
    
    # Compute loss
    loss = jnp.mean((pred_ys.squeeze() - ys.squeeze()) ** 2) + \
        args["lamb"] * (args["lamb_l1"] * l1_loss + args["lamb_entropy"] * entropy_loss)
    # Return loss and a dictionary of metrics here, along with the state
    return loss, ({"loss": loss, "batch_size": xs.shape[0]}, state)