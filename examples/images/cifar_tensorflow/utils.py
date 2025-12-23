"""
TensorFlow utilities for CIFAR-10 training.
Includes EMA, sample generation, and class-conditioned sampling.

Authors: Kilian Fatras
         Alexander Tong
         (TensorFlow port)
"""

import os
import numpy as np
import tensorflow as tf
from typing import Optional, Dict, Any


def ema_update(source_model, target_model, decay: float):
    """
    Update target model weights with exponential moving average of source model.
    
    Parameters
    ----------
    source_model : keras.Model
        The source model (updated during training)
    target_model : keras.Model
        The EMA model to update
    decay : float
        EMA decay rate (typically 0.9999)
    """
    for source_var, target_var in zip(source_model.trainable_variables, 
                                       target_model.trainable_variables):
        target_var.assign(decay * target_var + (1 - decay) * source_var)


class ExponentialMovingAverage:
    """
    Maintains exponential moving averages of model weights.
    """
    
    def __init__(self, model, decay: float = 0.9999):
        """
        Initialize EMA tracker.
        
        Parameters
        ----------
        model : keras.Model
            The model whose weights to track
        decay : float
            EMA decay rate
        """
        self.decay = decay
        self.shadow = [tf.Variable(v, trainable=False) for v in model.trainable_variables]
    
    def update(self, model):
        """Update EMA weights."""
        for shadow_var, model_var in zip(self.shadow, model.trainable_variables):
            shadow_var.assign(self.decay * shadow_var + (1 - self.decay) * model_var)
    
    def apply(self, model):
        """Apply EMA weights to model (for evaluation)."""
        self.backup = [tf.identity(v) for v in model.trainable_variables]
        for model_var, shadow_var in zip(model.trainable_variables, self.shadow):
            model_var.assign(shadow_var)
    
    def restore(self, model):
        """Restore original weights to model."""
        for model_var, backup_var in zip(model.trainable_variables, self.backup):
            model_var.assign(backup_var)


def euler_integrate(model, x0, num_steps=100):
    """
    Euler integration of the ODE from t=0 to t=1.
    
    Parameters
    ----------
    model : keras.Model
        Vector field model v(t, x)
    x0 : tf.Tensor
        Initial condition at t=0
    num_steps : int
        Number of integration steps
    
    Returns
    -------
    tf.Tensor
        Solution at t=1
    """
    dt = 1.0 / num_steps
    x = x0
    
    for i in range(num_steps):
        t = tf.fill([tf.shape(x)[0]], i * dt)
        v = model(t, x, training=False)
        x = x + dt * v
    
    return x


def rk4_integrate(model, x0, num_steps=100):
    """
    4th order Runge-Kutta integration of the ODE from t=0 to t=1.
    
    Parameters
    ----------
    model : keras.Model
        Vector field model v(t, x)
    x0 : tf.Tensor
        Initial condition at t=0
    num_steps : int
        Number of integration steps
    
    Returns
    -------
    tf.Tensor
        Solution at t=1
    """
    dt = 1.0 / num_steps
    x = x0
    
    for i in range(num_steps):
        t = i * dt
        t_tensor = tf.fill([tf.shape(x)[0]], t)
        t_mid = tf.fill([tf.shape(x)[0]], t + 0.5 * dt)
        t_end = tf.fill([tf.shape(x)[0]], t + dt)
        
        k1 = model(t_tensor, x, training=False)
        k2 = model(t_mid, x + 0.5 * dt * k1, training=False)
        k3 = model(t_mid, x + 0.5 * dt * k2, training=False)
        k4 = model(t_end, x + dt * k3, training=False)
        
        x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    
    return x


@tf.function
def generate_samples_batch(model, batch_size, num_steps=100, method='euler'):
    """
    Generate a batch of samples using ODE integration.
    
    Parameters
    ----------
    model : keras.Model
        Vector field model
    batch_size : int
        Number of samples to generate
    num_steps : int
        Number of integration steps
    method : str
        Integration method ('euler' or 'rk4')
    
    Returns
    -------
    tf.Tensor
        Generated samples, shape [batch_size, H, W, C]
    """
    # Start from Gaussian noise (NHWC format)
    x0 = tf.random.normal([batch_size, 32, 32, 3])
    
    if method == 'euler':
        samples = euler_integrate(model, x0, num_steps)
    elif method == 'rk4':
        samples = rk4_integrate(model, x0, num_steps)
    else:
        raise ValueError(f"Unknown integration method: {method}")
    
    return samples


def generate_and_save_samples(model, save_path, step, num_samples=64, 
                               num_steps=100, prefix="generated"):
    """
    Generate samples and save them as an image grid.
    
    Parameters
    ----------
    model : keras.Model
        Vector field model
    save_path : str
        Directory to save images
    step : int
        Current training step
    num_samples : int
        Number of samples to generate (should be a perfect square)
    num_steps : int
        Number of ODE integration steps
    prefix : str
        Prefix for saved file name
    """
    # Generate samples
    samples = generate_samples_batch(model, num_samples, num_steps)
    
    # Clip to valid range and convert to uint8
    samples = tf.clip_by_value(samples, -1.0, 1.0)
    samples = ((samples + 1.0) * 127.5).numpy().astype(np.uint8)
    
    # Create image grid
    grid_size = int(np.sqrt(num_samples))
    grid = np.zeros((grid_size * 32, grid_size * 32, 3), dtype=np.uint8)
    
    for i in range(grid_size):
        for j in range(grid_size):
            idx = i * grid_size + j
            if idx < num_samples:
                grid[i * 32:(i + 1) * 32, j * 32:(j + 1) * 32] = samples[idx]
    
    # Save using PIL or matplotlib
    try:
        from PIL import Image
        img = Image.fromarray(grid)
        os.makedirs(save_path, exist_ok=True)
        img.save(os.path.join(save_path, f"{prefix}_step_{step}.png"))
    except ImportError:
        import matplotlib.pyplot as plt
        os.makedirs(save_path, exist_ok=True)
        plt.imsave(os.path.join(save_path, f"{prefix}_step_{step}.png"), grid)


class ClassConditionedSampler:
    """
    A sampler that provides batches of samples from the same class.
    
    Original classes are divided into K subclasses using K-means clustering on the
    flattened image features. Each clustered subclass becomes its own class, so
    the total number of classes is num_classes * num_subclasses.
    This enables finer-grained locality training where samples come from the 
    same semantic cluster.
    """
    
    def __init__(self, images, labels, batch_size, num_classes=10, num_subclasses=1,
                 kmeans_max_iter=100, random_state=42, verbose=True):
        """
        Initialize the class-conditioned sampler.
        
        Parameters
        ----------
        images : np.ndarray or tf.Tensor
            Dataset images, shape [N, H, W, C]
        labels : np.ndarray or tf.Tensor
            Dataset labels, shape [N]
        batch_size : int
            Number of samples per batch
        num_classes : int
            Number of original classes in the dataset (default: 10 for CIFAR-10)
        num_subclasses : int
            Number of subclasses (K) to create within each original class.
            If 1, no clustering is performed.
            Total classes will be num_classes * num_subclasses.
        kmeans_max_iter : int
            Maximum iterations for K-means (default: 100)
        random_state : int
            Random seed for K-means reproducibility
        verbose : bool
            Print progress during clustering
        """
        self.images = np.array(images) if not isinstance(images, np.ndarray) else images
        self.labels = np.array(labels) if not isinstance(labels, np.ndarray) else labels
        self.batch_size = batch_size
        self.num_original_classes = num_classes
        self.num_subclasses = num_subclasses
        
        # Organize indices by original class
        original_class_to_indices = {}
        for class_id in range(num_classes):
            original_class_to_indices[class_id] = np.where(self.labels == class_id)[0]
        
        # Perform K-means clustering and create flattened class structure
        if num_subclasses > 1:
            if verbose:
                print(f"Performing K-means clustering with K={num_subclasses} for each of {num_classes} classes...")
            self.class_to_indices = {}
            self._perform_kmeans_clustering(original_class_to_indices, kmeans_max_iter, random_state, verbose)
        else:
            # No clustering, each original class becomes a single new class
            self.class_to_indices = {i: original_class_to_indices[i] for i in range(num_classes)}
        
        # Total number of classes after clustering
        self.num_classes = len(self.class_to_indices)
        
        # Shuffle indices for each class
        for class_id in self.class_to_indices.keys():
            np.random.shuffle(self.class_to_indices[class_id])
        
        # Track position in each class
        self.class_positions = {class_id: 0 for class_id in self.class_to_indices.keys()}
        
        if verbose:
            print(f"Created {self.num_classes} total classes after clustering")
            for class_id in sorted(self.class_to_indices.keys()):
                print(f"  Class {class_id}: {len(self.class_to_indices[class_id])} samples")
    
    def _perform_kmeans_clustering(self, original_class_to_indices, max_iter, random_state, verbose):
        """
        Perform K-means clustering within each original class and create flattened class IDs.
        
        Each cluster becomes a new class with ID: original_class_id * num_subclasses + subclass_id
        
        Uses sklearn's MiniBatchKMeans for efficiency with large datasets.
        """
        try:
            from sklearn.cluster import MiniBatchKMeans
        except ImportError:
            raise ImportError(
                "scikit-learn is required for K-means clustering. "
                "Install it with: pip install scikit-learn"
            )
        
        for original_class_id in range(self.num_original_classes):
            class_indices = original_class_to_indices[original_class_id]
            num_samples = len(class_indices)
            
            if verbose:
                print(f"  Clustering original class {original_class_id} ({num_samples} samples)...")
            
            # Load all images for this class and flatten them
            features = self.images[class_indices].reshape(num_samples, -1)
            
            # Adjust K if there are fewer samples than subclasses
            effective_k = min(self.num_subclasses, num_samples)
            
            if effective_k < self.num_subclasses and verbose:
                print(f"    Warning: Original class {original_class_id} has only {num_samples} samples, "
                      f"using K={effective_k} instead of {self.num_subclasses}")
            
            # Perform K-means clustering
            kmeans = MiniBatchKMeans(
                n_clusters=effective_k,
                max_iter=max_iter,
                random_state=random_state,
                batch_size=min(1024, num_samples),
                n_init='auto'
            )
            cluster_labels = kmeans.fit_predict(features)
            
            # Organize indices by new flattened class ID
            for subclass_id in range(effective_k):
                mask = cluster_labels == subclass_id
                subclass_indices = class_indices[mask]
                # Flatten: new class_id = original_class_id * num_subclasses + subclass_id
                new_class_id = original_class_id * self.num_subclasses + subclass_id
                self.class_to_indices[new_class_id] = subclass_indices
    
    def sample_batch_from_class(self, class_id=None):
        """
        Sample a batch of images all from the same class.
        
        Parameters
        ----------
        class_id : int, optional
            Specific class to sample from. If None, randomly select a class.
        
        Returns
        -------
        batch : tf.Tensor
            Batch of images, shape [batch_size, H, W, C]
        class_id : int
            The class that was sampled
        """
        if class_id is None:
            class_id = list(self.class_to_indices.keys())[
                np.random.randint(0, self.num_classes)
            ]
        
        class_indices = self.class_to_indices[class_id]
        current_pos = self.class_positions[class_id]
        
        # Reshuffle if needed
        if current_pos + self.batch_size > len(class_indices):
            np.random.shuffle(self.class_to_indices[class_id])
            self.class_positions[class_id] = 0
            current_pos = 0
        
        # Get batch indices
        batch_indices = class_indices[current_pos:current_pos + self.batch_size]
        self.class_positions[class_id] = current_pos + self.batch_size
        
        # Load data
        batch = self.images[batch_indices]
        batch = tf.constant(batch, dtype=tf.float32)
        
        return batch, class_id
    
    def __iter__(self):
        """Iterator that yields batches indefinitely."""
        while True:
            batch, class_id = self.sample_batch_from_class()
            yield batch, class_id


def load_cifar10(data_dir=None, normalize=True):
    """
    Load CIFAR-10 dataset.
    
    Parameters
    ----------
    data_dir : str, optional
        Directory to store/load data
    normalize : bool
        Whether to normalize to [-1, 1]
    
    Returns
    -------
    (x_train, y_train), (x_test, y_test)
    """
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    
    # Squeeze labels
    y_train = y_train.squeeze()
    y_test = y_test.squeeze()
    
    # Convert to float and normalize
    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)
    
    if normalize:
        x_train = (x_train - 127.5) / 127.5  # Normalize to [-1, 1]
        x_test = (x_test - 127.5) / 127.5
    
    return (x_train, y_train), (x_test, y_test)


def create_dataset(images, labels, batch_size, shuffle=True, augment=True):
    """
    Create a tf.data.Dataset from images and labels.
    
    Parameters
    ----------
    images : np.ndarray
        Images array
    labels : np.ndarray
        Labels array
    batch_size : int
        Batch size
    shuffle : bool
        Whether to shuffle the dataset
    augment : bool
        Whether to apply data augmentation
    
    Returns
    -------
    tf.data.Dataset
    """
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=10000)
    
    if augment:
        def augment_fn(image, label):
            # Random horizontal flip
            image = tf.image.random_flip_left_right(image)
            return image, label
        
        dataset = dataset.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
    
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


@tf.function
def compute_local_loss(net_model, x1, x0, sigma=0.0):
    """
    Compute the local regularization loss.
    
    This measures the locality property: ||v(u_t, t) - v(s_t, t) + s - u||²
    where u and s are samples from the same class/batch.
    
    Parameters
    ----------
    net_model : keras.Model
        The neural network model (vector field)
    x1 : tf.Tensor
        Target samples (should be from the same class)
    x0 : tf.Tensor
        Source samples (noise)
    sigma : float
        Noise standard deviation
    
    Returns
    -------
    tf.Tensor
        The computed local loss value
    """
    batch_size = tf.shape(x1)[0]
    
    if batch_size <= 1:
        return tf.constant(0.0)
    
    # Randomly pair samples within the batch
    perm = tf.random.shuffle(tf.range(batch_size))
    u = x1  # First set of samples
    s = tf.gather(x1, perm)  # Permuted samples
    
    # Sample the same time t for both u and s
    t_reg = tf.random.uniform([batch_size])
    t_expanded = tf.reshape(t_reg, [-1, 1, 1, 1])
    
    # Compute u_t and s_t with the same noise and time
    eps = tf.random.normal(tf.shape(x0))
    mu_u_t = t_expanded * u + (1 - t_expanded) * x0
    mu_s_t = t_expanded * s + (1 - t_expanded) * x0
    
    # Add same noise to both (if sigma > 0)
    u_t = mu_u_t + sigma * eps
    s_t = mu_s_t + sigma * eps
    
    # Compute v(u_t, t) and v(s_t, t)
    v_ut = net_model(t_reg, u_t, training=True)
    v_st = net_model(t_reg, s_t, training=True)
    
    # Local loss: ||v(u_t, t) - v(s_t, t) + s - u||²
    loss_local = tf.reduce_mean((v_ut - v_st + s - u) ** 2)
    
    return loss_local


def save_checkpoint(model, ema_model, optimizer, step, save_dir, name):
    """
    Save model checkpoint.
    
    Parameters
    ----------
    model : keras.Model
        Main model
    ema_model : keras.Model
        EMA model
    optimizer : keras.optimizers.Optimizer
        Optimizer
    step : int
        Current training step
    save_dir : str
        Directory to save checkpoint
    name : str
        Checkpoint name prefix
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Save model weights
    model.save_weights(os.path.join(save_dir, f"{name}_model_step_{step}.h5"))
    ema_model.save_weights(os.path.join(save_dir, f"{name}_ema_step_{step}.h5"))
    
    # Save optimizer state
    np.save(os.path.join(save_dir, f"{name}_optimizer_step_{step}.npy"), 
            optimizer.get_weights())
    
    # Save step
    with open(os.path.join(save_dir, f"{name}_step.txt"), 'w') as f:
        f.write(str(step))
    
    print(f"Checkpoint saved at step {step}")


def load_checkpoint(model, ema_model, optimizer, save_dir, name):
    """
    Load model checkpoint.
    
    Parameters
    ----------
    model : keras.Model
        Main model
    ema_model : keras.Model
        EMA model
    optimizer : keras.optimizers.Optimizer
        Optimizer
    save_dir : str
        Directory containing checkpoint
    name : str
        Checkpoint name prefix
    
    Returns
    -------
    int
        Training step to resume from
    """
    # Find latest step
    step_file = os.path.join(save_dir, f"{name}_step.txt")
    if not os.path.exists(step_file):
        return 0
    
    with open(step_file, 'r') as f:
        step = int(f.read().strip())
    
    # Load weights
    model.load_weights(os.path.join(save_dir, f"{name}_model_step_{step}.h5"))
    ema_model.load_weights(os.path.join(save_dir, f"{name}_ema_step_{step}.h5"))
    
    # Load optimizer state (optional, may fail if optimizer structure changed)
    try:
        opt_weights = np.load(os.path.join(save_dir, f"{name}_optimizer_step_{step}.npy"), 
                             allow_pickle=True)
        optimizer.set_weights(opt_weights)
    except Exception as e:
        print(f"Warning: Could not load optimizer state: {e}")
    
    print(f"Loaded checkpoint from step {step}")
    return step


class WarmupSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Learning rate schedule with linear warmup."""
    
    def __init__(self, target_lr, warmup_steps):
        super().__init__()
        self.target_lr = target_lr
        self.warmup_steps = warmup_steps
    
    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup = tf.cast(self.warmup_steps, tf.float32)
        return self.target_lr * tf.minimum(step / warmup, 1.0)
    
    def get_config(self):
        return {
            "target_lr": self.target_lr,
            "warmup_steps": self.warmup_steps,
        }


def visualize_clusters(class_sampler, output_dir="./group", samples_per_class=64, grid_nrow=8):
    """
    Visualize the clustering results by saving sample images from each cluster.
    
    Creates a grid of images for each cluster/class, showing what samples
    were grouped together by the K-means algorithm.
    
    Parameters
    ----------
    class_sampler : ClassConditionedSampler
        The sampler with class_to_indices mapping after clustering
    output_dir : str
        Directory where visualization images will be saved (default: "./group")
    samples_per_class : int
        Number of sample images to save per cluster (default: 64)
    grid_nrow : int
        Number of images per row in the grid (default: 8)
    
    Returns
    -------
    None
        Images are saved to disk in the output_dir
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Visualizing {class_sampler.num_classes} clusters in '{output_dir}/'...")
    
    # Determine original class info if clustering was performed
    if class_sampler.num_subclasses > 1:
        print(f"  (Clustered from {class_sampler.num_original_classes} original classes "
              f"with {class_sampler.num_subclasses} subclasses each)")
    
    # Save samples from each cluster
    for class_id in sorted(class_sampler.class_to_indices.keys()):
        class_indices = class_sampler.class_to_indices[class_id]
        num_samples = min(samples_per_class, len(class_indices))
        
        # Collect sample images from this cluster
        images = class_sampler.images[class_indices[:num_samples]]
        
        # Denormalize from [-1, 1] to [0, 255]
        images = ((images + 1.0) * 127.5).astype(np.uint8)
        
        # Create image grid
        grid_size = int(np.ceil(np.sqrt(num_samples)))
        grid = np.ones((grid_size * 32, grid_size * 32, 3), dtype=np.uint8) * 255
        
        for i in range(num_samples):
            row = i // grid_size
            col = i % grid_size
            grid[row * 32:(row + 1) * 32, col * 32:(col + 1) * 32] = images[i]
        
        # Determine filename
        if class_sampler.num_subclasses > 1:
            original_class_id = class_id // class_sampler.num_subclasses
            subclass_id = class_id % class_sampler.num_subclasses
            filename = f"class_{class_id:03d}_orig_{original_class_id}_sub_{subclass_id:02d}.png"
        else:
            filename = f"class_{class_id:03d}.png"
        
        # Save using PIL or matplotlib
        save_path = os.path.join(output_dir, filename)
        try:
            from PIL import Image
            img = Image.fromarray(grid)
            img.save(save_path)
        except ImportError:
            import matplotlib.pyplot as plt
            plt.imsave(save_path, grid)
        
        if (class_id + 1) % 10 == 0 or class_id == class_sampler.num_classes - 1:
            print(f"  Saved {class_id + 1}/{class_sampler.num_classes} cluster visualizations")
    
    print(f"Cluster visualization complete! Images saved to '{output_dir}/'")
    
    # Create a summary file
    summary_path = os.path.join(output_dir, "README.txt")
    with open(summary_path, 'w') as f:
        f.write("Cluster Visualization Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total clusters: {class_sampler.num_classes}\n")
        f.write(f"Original classes: {class_sampler.num_original_classes}\n")
        f.write(f"Subclasses per original class: {class_sampler.num_subclasses}\n")
        f.write(f"Samples per cluster visualization: {samples_per_class}\n\n")
        f.write("Each image grid shows samples from the same cluster.\n")
        if class_sampler.num_subclasses > 1:
            f.write("Filename format: class_XXX_orig_Y_sub_ZZ.png\n")
            f.write("  - XXX: new cluster ID (0 to {})\n".format(class_sampler.num_classes - 1))
            f.write("  - Y: original CIFAR-10 class (0-9)\n")
            f.write("  - ZZ: subcluster within original class (0 to {})\n".format(class_sampler.num_subclasses - 1))
        else:
            f.write("Filename format: class_XXX.png\n")
            f.write("  - XXX: class ID\n")
        f.write("\nCIFAR-10 Class Labels:\n")
        f.write("  0: airplane\n")
        f.write("  1: automobile\n")
        f.write("  2: bird\n")
        f.write("  3: cat\n")
        f.write("  4: deer\n")
        f.write("  5: dog\n")
        f.write("  6: frog\n")
        f.write("  7: horse\n")
        f.write("  8: ship\n")
        f.write("  9: truck\n")
    
    print(f"Summary saved to '{summary_path}'")