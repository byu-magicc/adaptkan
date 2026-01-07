from adaptkan.jax.losses import mse_loss
from tqdm import tqdm
import os
import io
import imageio
import matplotlib.pyplot as plt
import jax.numpy as jnp
import optax
import equinox as eqx
from adaptkan.jax.viz import plot_layer, render_animations, plot_layer_from_weights
from adaptkan.jax.data import DataLoader, weighted_average_metrics
import jax
from adaptkan.jax.model import AdaptKANJax
from jax import lax
from typing import Callable, Any, Dict, Optional, Union
from adaptkan.common.image_util import resize_images_to_common_size

# FIXME This doesn't quite work yet
def manual_adapt_interval(model, state, step, n):
    return lax.cond(
        jnp.mod(step, n) == 0,
        lambda _: model.manual_adapt(state),
        lambda _: (model, state, jnp.array(False)),
        operand=None
    )

@eqx.filter_jit
def train_step(
    model: eqx.Module,
    state: Any,
    opt_state: optax.OptState,
    batch: Dict[str, jnp.ndarray],
    loss_fn: Callable[..., Any],
    loss_args: Dict[str, Any],
    optim: optax.GradientTransformation,
    use_lbfgs: bool = False,
) -> Any:
    """Perform one optimization step on the entire batch."""

    (loss, (metrics, state)), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model, state, batch, loss_args)

    # Update model parameters
    if use_lbfgs:
        # Create a simplified loss function for LBFGS line search
        # Only pass the model parameters that will be optimized
        def value_fn(params):
            # Reconstruct the full model with new parameters
            model_updated = eqx.combine(params, eqx.filter(model, eqx.is_array, inverse=True))
            # Use the original state and batch, but set update=False
            loss_args_eval = {**loss_args, "update": False}
            loss_val, _ = loss_fn(model_updated, state, batch, loss_args_eval)
            return loss_val
        
        # Get only the array parameters for LBFGS
        params = eqx.filter(model, eqx.is_array)
        
        updates, opt_state = optim.update(
            grads,
            opt_state,
            params=params,
            value=loss,
            grad=grads,
            value_fn=value_fn
        )
    else:
        updates, opt_state = optim.update(grads, opt_state, params=eqx.filter(model, eqx.is_array))

    model = eqx.apply_updates(model, updates)
    return model, state, opt_state, metrics

def _prepare_data_loader(data, batch_size, seed, shuffle=True):
    """Convert various data formats to DataLoader"""
    if isinstance(data, DataLoader):
        return data
    elif isinstance(data, dict):
        loader_key = jax.random.PRNGKey(seed)
        return DataLoader(data, batch_size=batch_size, key=loader_key, shuffle=shuffle)
    else:
        raise ValueError("Data must be either a dictionary or DataLoader instance")

def fit(
    model: eqx.Module,
    state: Any,
    train_data: Union[Dict[str, jnp.ndarray], DataLoader, Any],
    test_data: Optional[Union[Dict[str, jnp.ndarray], DataLoader, Any]] = None,
    loss_fn: Callable[..., Any] = mse_loss,
    test_loss_fn: Optional[Callable[..., Any]] = None,
    opt: str = "Adam",
    learning_rate: float = 1.0,
    steps: int = 100,
    lamb: float = 0.0,
    log_freq: int = 1,
    display_metrics: Optional[list] = None,
    batch_size: int = -1,
    snapshot_every: Optional[int] = None,
    save_name: str = "simple_equation",
    save_path: str = "results/animations",
    fps: int = 10,
    loss_args: Optional[dict] = None,
    adapt_model: bool = True,
    seed: int = 0,
    start_steps: Optional[int] = 0,
    preset_optim: Optional[optax.GradientTransformation] = None,
    preset_opt_state: Optional[optax.OptState] = None,
    return_optim: bool = False,
    save_individual_frames: bool = False,
):
    """
    Fit the model to the training data using the specified loss function and optimizer.

    Args:
        model (eqx.Module): The model to be trained.
        state (Any): The initial state of the model.
        train_data (Union[Dict[str, jnp.ndarray], DataLoader, Any]): Training data, either as a dictionary or DataLoader.
        test_data (Optional[Union[Dict[str, jnp.ndarray], DataLoader, Any]]): Optional test data for evaluation.
        loss_fn (Callable[..., Any]): Loss function to be used for training.
        test_loss_fn (Optional[Callable[..., Any]]): Optional loss function for testing.
        opt (str): Optimizer to be used for training. Options include "SGD", "Adam", "AdamW", and "LBFGS".
        learning_rate (float): Learning rate for the optimizer.
        steps (int): Number of training steps.
        lamb (float): Regularization parameter for weight decay.
        log_freq (int): Frequency of logging metrics during training.
        display_metrics (Optional[list]): List of metrics to display during training.
        batch_size (int): Size of each training batch. If -1, use the full dataset as a batch.
        snapshot_every (Optional[int]): Frequency (in steps) of saving snapshots for visualization.
        save_name (str): Name for saving snapshots and animations.
        save_path (str): Path for saving snapshots and animations.
        fps (int): Frames per second for saved animations.
        loss_args (Optional[dict]): Additional arguments for the loss function.
        adapt_model (bool): Whether to adapt the model during training.
        seed (int): Random seed for reproducibility.
        start_steps (Optional[int]): Starting step for training, useful for resuming.
        preset_optim (Optional[optax.GradientTransformation]): Predefined optimizer to use.
        preset_opt_state (Optional[optax.OptState]): Predefined optimizer state to use.
        return_optim (bool): Whether to return the optimizer and its state.
        save_individual_frames (bool): Whether to save individual frames of animations.

    Returns:
        Tuple: Trained model, final state, and training results (metrics). Optionally returns the optimizer and its state if `return_optim` is True.
    """

    end_steps = start_steps + steps

    # Prepare
    if loss_args is None:
        loss_args = {}
    if test_loss_fn is None:
        test_loss_fn = loss_fn

    if "update" not in loss_args:
        loss_args["update"] = True

    # Build optimizer once
    if preset_optim is None:
        if opt == "SGD":
            optim = optax.chain(optax.add_decayed_weights(lamb), optax.sgd(learning_rate))
        elif opt == "Adam":
            optim = optax.chain(optax.add_decayed_weights(lamb), optax.adam(learning_rate))
        elif opt == "AdamW":
            schedule = optax.polynomial_schedule(
                init_value=learning_rate,
                end_value=learning_rate / 10,
                power=2,  # Quadratic decay
                transition_steps=2000
            )
            optim = optax.chain(optax.add_decayed_weights(lamb), optax.adamw(schedule, b1=0.99, weight_decay=1e-5, nesterov=True))
        elif opt == "LBFGS":
            # linesearch = optax.scale_by_backtracking_linesearch(max_backtracking_steps=15)
            # linesearch = optax.scale_by_zoom_linesearch(max_linesearch_steps=15) 
            # optim = optax.chain(optax.lbfgs(), linesearch)
            # Try increasing the number of linesearch steps
            # optim = optax.lbfgs()
            optim = optax.lbfgs(
                learning_rate=1.0,              # PyTorch lr=1
                memory_size=10,                # KAN history_size=10
                linesearch=optax.scale_by_zoom_linesearch(
                    max_linesearch_steps=25,    # PyTorch max_eval=25 (max_iter * 1.25)
                    slope_rtol=1e-4,            # PyTorch c1=1e-4 (Armijo condition)
                    curv_rtol=0.9,              # PyTorch c2=0.9 (strong curvature condition)
                    stepsize_precision=1e-32,   # KAN tolerance_change=1e-9
                    verbose=False               # Set True for debugging
                )
            )
            # optim = optax.lbfgs(scale_init_precond=False,
            #                     linesearch=optax.scale_by_zoom_linesearch(max_linesearch_steps=20,
            #                                                               verbose=True,
            #                                                               initial_guess_strategy='one'))
        else:
            raise ValueError(f"opt {opt} is not supported.")
        opt_state = optim.init(eqx.filter(model, eqx.is_array))
    else:
        # If we already have a preset optimizer and state we are going from, use those
        optim = preset_optim
        opt_state = preset_opt_state

    # Get dataset size first to determine proper batch size
    if isinstance(train_data, DataLoader):
        num_train = train_data.dataset_size
        effective_batch_size = train_data.batch_size
    elif isinstance(train_data, dict):
        num_train = next(iter(train_data.values())).shape[0]
        # Handle batch_size=-1 (full batch) or invalid batch sizes
        if batch_size < 1 or batch_size > num_train:
            effective_batch_size = num_train
        else:
            effective_batch_size = batch_size
    else:
        raise ValueError("Data must be either a dictionary or DataLoader instance")
    
    # Validate effective_batch_size
    if effective_batch_size <= 0:
        raise ValueError(f"Effective batch size must be positive, got {effective_batch_size}")
    if effective_batch_size > num_train:
        print(f"Warning: effective_batch_size ({effective_batch_size}) > dataset_size ({num_train})")
        effective_batch_size = num_train

    # Prepare data loaders with corrected batch size
    shuffle = effective_batch_size != num_train

    # Create a train loader
    train_loader = _prepare_data_loader(train_data, effective_batch_size, seed, shuffle=shuffle)
    if display_metrics is None:
        display_metrics = ["train_loss"]
        if test_data is not None:
            display_metrics.append("test_loss")

    # Create our test loader
    test_loader = None
    if test_data is not None:
        if isinstance(test_data, dict):
            test_num = next(iter(test_data.values())).shape[0]
            test_batch_size = effective_batch_size if effective_batch_size <= test_num else test_num
        else:
            test_batch_size = effective_batch_size
        test_loader = _prepare_data_loader(test_data, test_batch_size, seed+1, shuffle=False)

    # Get dataset size for progress tracking
    if hasattr(train_loader, 'dataset_size'):
        num_train = train_loader.dataset_size
    else:
        # Fallback for legacy format
        if isinstance(train_data, dict):
            num_train = next(iter(train_data.values())).shape[0]
        else:
            num_train = train_data.shape[0]
        
    if batch_size < 1 or batch_size > num_train:
        batch_size = num_train

    # Setup animation buffers
    if snapshot_every is not None:
        num_layers = len(model.layers)
        frames_per_layer = [[] for _ in range(num_layers)]

    # Results
    results = {key: [] for key in display_metrics}
    results["train_loss"] = []
    if test_loader is not None:
        results["test_loss"] = []

    # Main loop
    pbar = tqdm(range(start_steps, end_steps), desc="Training", ncols=100)

    adapt_first = loss_args.get("adapt_first", False)
    adapt_first_epoch_only = loss_args.get("adapt_first_epoch_only", False)
    
    adapted_epochs = []
    step_counter = 0
    for epoch in pbar:
        adapted = False
        all_metrics = []

        for batch_idx, batch in enumerate(train_loader):

            if hasattr(model, "adapt") and batch_idx == 0 and adapt_first and adapt_model:
                if (epoch > 0 and not adapt_first_epoch_only) or epoch == 0: 
                    if loss_args.get("manual_adapt_every", None) is not None:
                        if epoch % loss_args["manual_adapt_every"] == 0:
                            model, state, adapted = model.manual_adapt(state)
                    else:
                        model, state, adapted = model.adapt(state)

            # _JITted_ train step on full micro‑batch
            model, state, opt_state, metrics = train_step(
                model,
                state,
                opt_state,
                batch,
                loss_fn,
                {**loss_args, "current_epoch": jnp.array(epoch)},
                optim,
                use_lbfgs=opt=="LBFGS",
            )
            
            if hasattr(model, "adapt") and batch_idx == 0 and not adapt_first and adapt_model:
                if (epoch > 0 and not adapt_first_epoch_only) or epoch == 0: 
                    if loss_args.get("manual_adapt_every", None) is not None:
                        if epoch % loss_args["manual_adapt_every"] == 0:
                            model, state, adapted = model.manual_adapt(state)
                    else:
                        model, state, adapted = model.adapt(state)

            if adapted:
                adapted_epochs.append(epoch)

            # Extract loss and batch size from metrics
            if "batch_size" not in metrics:
                raise ValueError("train_step must return 'batch_size' in metrics for proper averaging")
            
            all_metrics.append(metrics)
            step_counter += 1

        batch_sizes = [m["batch_size"] for m in all_metrics]
        metrics = weighted_average_metrics(all_metrics, batch_sizes)
        metrics = {f"train_{k}": v for k, v in metrics.items()}

        # Run test loader
        # Evaluate test loss if provided
        if test_loader is not None:
            test_batch_sizes = []

            for test_batch in test_loader:
                all_test_metrics = []
                test_loss, (test_metrics, _) = test_loss_fn(
                    model, state, test_batch, {**loss_args, "update": False, "index_batch": False}
                )

                # Extract batch size from test metrics too
                if "batch_size" not in test_metrics:
                    raise ValueError("test_loss_fn must return 'batch_size' in metrics for proper averaging")

                all_test_metrics.append(test_metrics)

            test_batch_sizes = [m["batch_size"] for m in all_test_metrics]
            test_metrics = weighted_average_metrics(all_test_metrics, test_batch_sizes)
            
            test_metrics_prefixed = {f"test_{k}": v for k, v in test_metrics.items()}
            metrics = {**metrics, **test_metrics_prefixed}

        # Record results (still JAX arrays)
        [results[key].append(metrics[key].item()) for key in metrics if key in results]

        if epoch % log_freq == 0:
            if display_metrics == None:
                if metrics["test_loss"] is None:
                    pbar.set_description(f"train_loss: {metrics['loss'].item():.8f} ")
                else:
                    pbar.set_description(f"train_loss: {metrics['loss'].item():.8f} | test_loss: {metrics['test_loss'].item():.8f} ")
            else:
                # TODO test that this works displaying different metrics
                string = ''
                data = ()
                for metric in display_metrics:
                    string += f' {metric}: %.8f |'
                    try:
                        metrics[metric]
                    except:
                        raise Exception(f'{metric} not recognized')
                    data += (metrics[metric].item(),)
                pbar.set_description(string % data)

        # Collect frames (no plotting here)
        if snapshot_every is not None and epoch % snapshot_every == 0:
            # just store raw layer params for deferred rendering
            for idx, layer in enumerate(model.layers):
                item = layer.weights, state.get(layer.a), state.get(layer.b), state.get(layer.data_counts), state.get(layer.ood_data_counts)
                frames_per_layer[idx].append(item)  # simple example

    # Post‑train: render animations if requested
    if snapshot_every is not None:
        os.makedirs(os.path.join(save_path, save_name), exist_ok=True)
        # Convert raw frames to visuals in one go
        for layer_idx, raw_frames in enumerate(frames_per_layer):
            
            # This assumes that these parameters don't change
            num_grid_intervals = model.layers[layer_idx].num_grid_intervals
            k = model.layers[layer_idx].k
            rounding_precision_eps = model.layers[layer_idx].rounding_precision_eps
            basis_type = model.layers[layer_idx].basis_type

            images = [
                plot_layer_from_weights(weights,
                                        a,
                                        b,
                                        num_grid_intervals,
                                        data_counts,
                                        ood_data_counts,
                                        layer_idx,
                                        k,
                                        rounding_precision_eps,
                                        basis_type=basis_type)
                for weights, a, b, data_counts, ood_data_counts in raw_frames
            ]

            processed_images = resize_images_to_common_size(images, method='pad')

            imageio.mimsave(
                os.path.join(save_path, save_name, f"layer_{layer_idx+1}.gif"),
                processed_images,
                fps=fps,
            )

            if save_individual_frames:
                # Save individual frames
                frames_dir = os.path.join(save_path, save_name, f"layer_{layer_idx+1}_frames")
                os.makedirs(frames_dir, exist_ok=True)
                
                for frame_idx, image in enumerate(images):
                    frame_filename = os.path.join(frames_dir, f"frame_{frame_idx:04d}.png")
                    imageio.imwrite(frame_filename, image)
                print(f"Saved GIF and {len(images)} frames for Layer {layer_idx+1}")
            else:
                print(f"Saved GIF and {len(images)} frames for Layer {layer_idx+1}")

    # print("cache size train step", train_step.__wrapped__._cache_size())
    jax.clear_caches()
    eqx.clear_caches()   # clears Equinox’s tree‐filter caches

    results['adapted_epochs'] = adapted_epochs

    if return_optim:
        return model, state, results, optim, opt_state
    else:
        return model, state, results

def evaluate(model,
             state,
             test_data,
             seed,
             batch_size=-1,
             test_loss_fn: Optional[Callable[..., Any]] = None,
             loss_args: Optional[dict] = None,):
    
    """
    Evaluate a model on test data and compute test metrics.
    Args:
        model: The model to be evaluated.
        state: The current state of the model, typically including parameters and optimizer state.
        test_data: The test dataset, which can be either a dictionary of arrays or a DataLoader instance.
        seed: Random seed for shuffling or data preparation.
        batch_size (int, optional): Batch size for evaluation. If -1 or invalid, the full dataset is used as a single batch. Defaults to -1.
        test_loss_fn (Callable[..., Any], optional): A callable loss function for evaluation. If None, defaults to `mse_loss`.
        loss_args (dict, optional): Additional arguments to pass to the loss function. Defaults to an empty dictionary.
    Returns:
        dict: A dictionary of test metrics with keys prefixed by "test_".
    Raises:
        ValueError: If `test_data` is neither a dictionary nor a DataLoader instance.
        ValueError: If `test_loss_fn` does not return a "batch_size" key in the metrics.
    """

    if isinstance(test_data, DataLoader):
        num_train = test_data.dataset_size
        effective_batch_size = test_data.batch_size
    elif isinstance(test_data, dict):
        num_train = next(iter(test_data.values())).shape[0]
        # Handle batch_size=-1 (full batch) or invalid batch sizes
        if batch_size < 1 or batch_size > num_train:
            effective_batch_size = num_train
        else:
            effective_batch_size = batch_size
    else:
        raise ValueError("Data must be either a dictionary or DataLoader instance")
    
    if isinstance(test_data, dict):
        test_num = next(iter(test_data.values())).shape[0]
        test_batch_size = effective_batch_size if effective_batch_size <= test_num else test_num
    else:
        test_batch_size = effective_batch_size
    test_loader = _prepare_data_loader(test_data, test_batch_size, seed, shuffle=False)    

    test_batch_sizes = []

    if loss_args is None:
        loss_args = {}

    if test_loss_fn is None:
        test_loss_fn = mse_loss

    for test_batch in test_loader:
        all_test_metrics = []
        test_loss, (test_metrics, _) = test_loss_fn(
            model, state, test_batch, {**loss_args, "update": False}
        )

        # Extract batch size from test metrics too
        if "batch_size" not in test_metrics:
            raise ValueError("test_loss_fn must return 'batch_size' in metrics for proper averaging")

        all_test_metrics.append(test_metrics)

    test_batch_sizes = [m["batch_size"] for m in all_test_metrics]
    test_metrics = weighted_average_metrics(all_test_metrics, test_batch_sizes)
    
    test_metrics_prefixed = {f"test_{k}": v for k, v in test_metrics.items()}

    return test_metrics_prefixed