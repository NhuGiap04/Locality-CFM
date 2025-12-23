import copy
import os

import torch
from torch import distributed as dist
from torchdyn.core import NeuralODE

# from torchvision.transforms import ToPILImage
from torchvision.utils import save_image

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def setup(
    rank: int,
    total_num_gpus: int,
    master_addr: str = "localhost",
    master_port: str = "12355",
    backend: str = "nccl",
):
    """Initialize the distributed environment.

    Args:
        rank: Rank of the current process.
        total_num_gpus: Number of GPUs used in the job.
        master_addr: IP address of the master node.
        master_port: Port number of the master node.
        backend: Backend to use.
    """
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port

    # initialize the process group
    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=total_num_gpus,
    )


def generate_samples(model, parallel, savedir, step, net_="normal"):
    """Save 64 generated images (8 x 8) for sanity check along training.

    Parameters
    ----------
    model:
        represents the neural network that we want to generate samples from
    parallel: bool
        represents the parallel training flag. Torchdyn only runs on 1 GPU, we need to send the models from several GPUs to 1 GPU.
    savedir: str
        represents the path where we want to save the generated images
    step: int
        represents the current step of training
    """
    model.eval()

    model_ = copy.deepcopy(model)
    if parallel:
        # Send the models from GPU to CPU for inference with NeuralODE from Torchdyn
        model_ = model_.module.to(device)

    node_ = NeuralODE(model_, solver="euler", sensitivity="adjoint")
    with torch.no_grad():
        traj = node_.trajectory(
            torch.randn(64, 3, 32, 32, device=device),
            t_span=torch.linspace(0, 1, 100, device=device),
        )
        traj = traj[-1, :].view([-1, 3, 32, 32]).clip(-1, 1)
        traj = traj / 2 + 0.5
    save_image(traj, savedir + f"{net_}_generated_FM_images_step_{step}.png", nrow=8)

    model.train()


def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(
            target_dict[key].data * decay + source_dict[key].data * (1 - decay)
        )


def infiniteloop(dataloader):
    while True:
        for x, y in iter(dataloader):
            yield x


class ClassConditionedSampler:
    """
    A sampler that provides batches of samples from the same class.
    
    Original classes are divided into K subclasses using K-means clustering on the
    flattened image features. Each clustered subclass becomes its own class, so
    the total number of classes is num_classes * num_subclasses.
    This enables finer-grained locality training where samples come from the 
    same semantic cluster.
    """
    
    def __init__(self, dataset, batch_size, num_classes=10, num_subclasses=1, 
                 kmeans_max_iter=100, random_state=42, verbose=True):
        """
        Args:
            dataset: PyTorch dataset with (image, label) pairs
            batch_size: Number of samples per batch
            num_classes: Number of original classes in the dataset (default: 10 for CIFAR-10)
            num_subclasses: Number of subclasses (K) to create within each original class.
                           If 1, no clustering is performed.
                           Total classes will be num_classes * num_subclasses.
            kmeans_max_iter: Maximum iterations for K-means (default: 100)
            random_state: Random seed for K-means reproducibility
            verbose: Print progress during clustering
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_original_classes = num_classes
        self.num_subclasses = num_subclasses
        
        # Organize dataset indices by original class
        original_class_to_indices = {i: [] for i in range(num_classes)}
        for idx, (_, label) in enumerate(dataset):
            original_class_to_indices[label].append(idx)
        
        # Convert to tensors for efficient indexing
        for class_id in range(num_classes):
            original_class_to_indices[class_id] = torch.tensor(original_class_to_indices[class_id])
        
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
        
        # Store the current position in each class
        self.class_positions = {class_id: 0 for class_id in self.class_to_indices.keys()}
        
        # Shuffle indices for each class
        for class_id in self.class_to_indices.keys():
            indices = self.class_to_indices[class_id]
            perm = torch.randperm(len(indices))
            self.class_to_indices[class_id] = indices[perm]
        
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
        
        import numpy as np
        
        for original_class_id in range(self.num_original_classes):
            class_indices = original_class_to_indices[original_class_id]
            num_samples = len(class_indices)
            
            if verbose:
                print(f"  Clustering original class {original_class_id} ({num_samples} samples)...")
            
            # Load all images for this class and flatten them
            features = []
            for idx in class_indices:
                img, _ = self.dataset[idx.item()]
                # Flatten image to 1D feature vector
                features.append(img.numpy().flatten())
            
            features = np.array(features)
            
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
                subclass_indices = class_indices[torch.tensor(mask)]
                # Flatten: new class_id = original_class_id * num_subclasses + subclass_id
                new_class_id = original_class_id * self.num_subclasses + subclass_id
                self.class_to_indices[new_class_id] = subclass_indices
    
    def sample_batch_from_class(self, class_id=None):
        """
        Sample a batch of images all from the same class.
        
        Args:
            class_id: Specific class to sample from. If None, randomly select a class.
        
        Returns:
            batch: Tensor of shape (batch_size, C, H, W)
            class_id: The class ID that was sampled
        """
        # Randomly select a class if not specified
        if class_id is None:
            class_id = list(self.class_to_indices.keys())[
                torch.randint(0, self.num_classes, (1,)).item()
            ]
        
        # Get indices for this class
        class_indices = self.class_to_indices[class_id]
        current_pos = self.class_positions[class_id]
        
        # If we don't have enough samples left in this class, reshuffle
        if current_pos + self.batch_size > len(class_indices):
            perm = torch.randperm(len(class_indices))
            self.class_to_indices[class_id] = class_indices[perm]
            class_indices = self.class_to_indices[class_id]
            current_pos = 0
            self.class_positions[class_id] = 0
        
        # Get batch indices
        batch_indices = class_indices[current_pos:current_pos + self.batch_size]
        self.class_positions[class_id] = current_pos + self.batch_size
        
        # Load the actual data
        batch_data = []
        for idx in batch_indices:
            img, _ = self.dataset[idx.item()]
            batch_data.append(img)
        
        batch = torch.stack(batch_data)
        
        return batch, class_id
    
    def __iter__(self):
        """Iterator that yields batches indefinitely."""
        while True:
            batch, class_id = self.sample_batch_from_class()
            yield batch, class_id


def compute_local_loss(net_model, x1, x0, sigma=0.0):
    """
    Compute the local regularization loss.
    
    This measures the locality property: ||v(u_t, t) - v(s_t, t) + s - u||^2
    where u and s are samples from the same class/batch.
    
    Parameters
    ----------
    net_model:
        The neural network model (vector field)
    x1: torch.Tensor
        Target samples (should be from the same class for meaningful local loss)
    x0: torch.Tensor
        Source samples (noise)
    sigma: float
        Noise standard deviation (default: 0.0)
        
    Returns
    -------
    torch.Tensor
        The computed local loss value
    """
    batch_size = x1.shape[0]
    
    if batch_size <= 1:
        return torch.tensor(0.0, device=x1.device)
    
    # Randomly pair samples within the batch (same class)
    perm = torch.randperm(batch_size, device=x1.device)
    u = x1  # First set of samples
    s = x1[perm]  # Permuted samples (different pairing)
    
    # Sample the same time t for both u and s
    t_reg = torch.rand(batch_size).type_as(x1)
    
    # Compute u_t and s_t with the same noise and time
    eps = torch.randn_like(x0)
    mu_u_t = t_reg.reshape(-1, 1, 1, 1) * u + (1 - t_reg.reshape(-1, 1, 1, 1)) * x0
    mu_s_t = t_reg.reshape(-1, 1, 1, 1) * s + (1 - t_reg.reshape(-1, 1, 1, 1)) * x0
    
    # Add same noise to both (if sigma > 0)
    u_t = mu_u_t + sigma * eps
    s_t = mu_s_t + sigma * eps
    
    # Compute v(u_t, t) and v(s_t, t)
    v_ut = net_model(t_reg, u_t)
    v_st = net_model(t_reg, s_t)
    
    # Local loss: ||v(u_t, t) - v(s_t, t) + s - u||^2
    loss_local = torch.mean((v_ut - v_st + s - u) ** 2)
    
    return loss_local


def visualize_clusters(class_sampler, dataset, output_dir="./group", samples_per_class=64, grid_nrow=8):
    """
    Visualize the clustering results by saving sample images from each cluster.
    
    Creates a grid of images for each cluster/class, showing what samples
    were grouped together by the K-means algorithm.
    
    Parameters
    ----------
    class_sampler: ClassConditionedSampler
        The sampler with class_to_indices mapping after clustering
    dataset: torch.utils.data.Dataset
        The dataset containing the images
    output_dir: str
        Directory where visualization images will be saved (default: "./group")
    samples_per_class: int
        Number of sample images to save per cluster (default: 64)
    grid_nrow: int
        Number of images per row in the grid (default: 8)
    
    Returns
    -------
    None
        Images are saved to disk in the output_dir
    """
    import os
    from torchvision.utils import save_image
    
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
        images = []
        for idx in class_indices[:num_samples]:
            img, original_label = dataset[idx.item()]
            images.append(img)
        
        # Stack images and normalize to [0, 1] range for visualization
        images_tensor = torch.stack(images)
        images_tensor = images_tensor / 2 + 0.5  # Denormalize from [-1, 1] to [0, 1]
        
        # Determine original class info for filename
        if class_sampler.num_subclasses > 1:
            original_class_id = class_id // class_sampler.num_subclasses
            subclass_id = class_id % class_sampler.num_subclasses
            filename = f"class_{class_id:03d}_orig_{original_class_id}_sub_{subclass_id:02d}.png"
        else:
            filename = f"class_{class_id:03d}.png"
        
        # Save grid of images
        save_path = os.path.join(output_dir, filename)
        save_image(images_tensor, save_path, nrow=grid_nrow, padding=2)
        
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
