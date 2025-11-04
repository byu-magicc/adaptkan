import jax
import jax.numpy as jnp

class DataLoader:
    def __init__(self, data_dict, batch_size, shuffle=True, key=None, drop_last=False):
        """
        Args:
            data_dict: Dictionary of JAX arrays, e.g. {'X': features, 'y': targets}
            batch_size: Size of each batch
            shuffle: Whether to shuffle data each epoch
            key: JAX random key for reproducible shuffling (optional)
            drop_last: Whether to drop the last incomplete batch
        """
        self.data_dict = data_dict
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.key = key if key is not None else jax.random.PRNGKey(42)
        self.drop_last = drop_last
        
        # Verify all arrays have same first dimension
        self.dataset_size = None
        for name, array in data_dict.items():
            if self.dataset_size is None:
                self.dataset_size = array.shape[0]
            else:
                assert array.shape[0] == self.dataset_size, f"Array '{name}' has different size"
        
        # Ensure we have data
        if self.dataset_size == 0:
            raise ValueError("Dataset is empty")
        
        # Adjust batch_size if it's larger than dataset
        if self.batch_size > self.dataset_size:
            print(f"Warning: batch_size ({self.batch_size}) > dataset_size ({self.dataset_size}). Setting batch_size = dataset_size.")
            self.batch_size = self.dataset_size
        
        self.indices = jnp.arange(self.dataset_size)
    
    def __iter__(self):
        """Create a new iterator for this epoch"""
        if self.shuffle:
            # Split key for this epoch to maintain reproducibility
            self.key, subkey = jax.random.split(self.key)
            perm = jax.random.permutation(subkey, self.indices)
        else:
            perm = self.indices
        
        start = 0
        while start < self.dataset_size:
            end = min(start + self.batch_size, self.dataset_size)
            
            # Skip incomplete batch if drop_last=True
            if self.drop_last and end - start < self.batch_size:
                break
            
            # Ensure we don't create empty batches
            if end <= start:
                break
                
            batch_indices = perm[start:end]
            
            # Create batch dictionary
            batch = {name: array[batch_indices] for name, array in self.data_dict.items()}
            
            # Debug: Print batch shapes
            # print(f"Batch start={start}, end={end}, indices shape={batch_indices.shape}")
            # for name, arr in batch.items():
            #     print(f"  {name}: {arr.shape}")
            
            yield batch
            
            start = end
    
    def __len__(self):
        """Number of batches per epoch"""
        if self.dataset_size == 0:
            return 0
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
            
        if self.drop_last:
            result = self.dataset_size // self.batch_size
        else:
            result = (self.dataset_size + self.batch_size - 1) // self.batch_size
        
        return max(0, result)  # Ensure non-negative
    
def weighted_average_metrics(epoch_metrics_list, batch_sizes):
    """
    Compute weighted average of all metrics across batches.
    
    Args:
        epoch_metrics_list: List of metric dictionaries from each batch
        batch_sizes: List of batch sizes corresponding to each metric dict
        
    Returns:
        Dictionary with weighted averages of all metrics
    """
    if len(epoch_metrics_list) == 1:
        # Single batch - no averaging needed
        return epoch_metrics_list[0]
    
    # Convert to arrays for efficient computation
    batch_sizes = jnp.array(batch_sizes)
    total_samples = jnp.sum(batch_sizes)
    
    # Get all metric names (excluding batch_size)
    all_metrics = set()
    for metrics in epoch_metrics_list:
        all_metrics.update(metrics.keys())
    all_metrics.discard('batch_size')  # Don't average batch_size itself
    
    # Compute weighted average for each metric
    weighted_metrics = {}
    for metric_name in all_metrics:
        # Extract metric values from all batches
        metric_values = jnp.array([metrics.get(metric_name, 0.0) for metrics in epoch_metrics_list])
        
        # Compute weighted average
        weighted_metrics[metric_name] = jnp.sum(metric_values * batch_sizes) / total_samples
    
    # Add total batch size for reference
    weighted_metrics['batch_size'] = total_samples
    
    return weighted_metrics