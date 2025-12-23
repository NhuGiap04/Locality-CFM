# CIFAR-10 Experiments using TorchCFM

This repository is used to reproduce the CIFAR-10 experiments from [Improving and generalizing flow-based generative models with minibatch optimal transport](https://arxiv.org/abs/2302.00482). We have designed a novel experimental procedure that helps us to reach an **FID of 3.5** on the CIFAR-10 dataset.

<p align="center">
<img src="../../../assets/169_generated_samples_otcfm.png" width="600"/>
</p>

## System Requirements

**GPU Memory Requirements:**
- **8GB VRAM**: Use `--batch_size=32` or `--batch_size=64` (recommended for testing)
- **11GB+ VRAM**: Use `--batch_size=128` (default, optimal for training)

The default batch size of 128 requires approximately 8GB of GPU memory, but with optimizer states and additional computations, at least 10-11GB is recommended.

## Training Scripts

### Standard Training (`train_cifar10.py`)

Trains standard Conditional Flow Matching models. Includes implicit locality monitoring by computing local loss on same-class samples for logging purposes only.

**OT-Conditional Flow Matching (OT-CFM):**
```bash
python train_cifar10.py --model=otcfm --lr=2e-4 --ema_decay=0.9999 --batch_size=128 --total_steps=400001 --save_step=20000 --num_subclasses=25 --visualize=True
```

**Independent Conditional Flow Matching (I-CFM):**
```bash
python train_cifar10.py --model=icfm --lr=2e-4 --ema_decay=0.9999 --batch_size=128 --total_steps=400001 --save_step=20000 --visualize=True
```

**Original Flow Matching (FM):**
```bash
python train_cifar10.py --model=fm --lr=2e-4 --ema_decay=0.9999 --batch_size=128 --total_steps=400001 --save_step=20000 --visualize=True
```

### Locality-Regularized Training (`train_cifar10_locality.py`)

Trains with explicit local regularization loss to enforce locality properties. Samples batches from the same class and adds a regularization term: λ · ||v(u_t, t) - v(s_t, t) + s - u||²

```bash
python train_cifar10_locality.py --model=otcfm --batch_size=128 --total_steps=400001 --save_step=20000 --lambda_local=1.0 --lambda_name=1e0 --num_subclasses=25 --visualize=True
```

**Key Parameters:**
- `--lambda_local`: Weight for local regularization term (default: 1.0)
- `--lambda_name`: String identifier for experiment naming (e.g., "1e0", "1e-1")
- `--num_subclasses`: Number of subclasses for finer-grained class-conditioned sampling (default: 25)
- `--visualize`: Visualize clustering results by saving sample images from each cluster (default: True)

### Anchor-Based Training (`train_cifar10_anchor.py`)

Trains with anchor-based regularization using class centroids to encourage locality. Computes centroids for each class and regularizes the vector field to be consistent between samples and their class centroids.

```bash
python train_cifar10_anchor.py --model=otcfm --batch_size=128 --total_steps=400001 --save_step=20000 --lambda_anchor=1.0 --lambda_name=1e0 --centroid_update_freq=1000 --anchor_loss_type=full --num_subclasses=25 --visualize=True
```

**Key Parameters:**
- `--lambda_anchor`: Weight for anchor regularization term (default: 1.0)
- `--lambda_name`: String identifier for experiment naming
- `--centroid_update_freq`: How often to recompute class centroids (default: 1000)
- `--anchor_loss_type`: Type of anchor loss - `full` or `simple` (default: "full")
- `--num_subclasses`: Number of subclasses for class-conditioned sampling (default: 25)
- `--visualize`: Visualize clustering results by saving sample images from each cluster (default: True)

## Logged Metrics

All training scripts log the following metrics to Weights & Biases every 100 steps:

- **`loss_fm`**: Standard flow matching loss (mean squared error between predicted and target velocity)
- **`regularized_loss`**: Local or anchor regularization loss (depending on training script)
- **`grad_norm`**: L2 norm of gradients before clipping (useful for monitoring training stability)
- **`vector_field_mean_norm`**: Mean L2 norm of predicted velocity vectors
- **`lr`**: Current learning rate
- **`step`**: Training step number
- **`fid`**: Fréchet Inception Distance (computed at checkpoint saves)

Additionally, the progress bar displays real-time metrics during training.

## Resuming from Checkpoints

All training scripts support resuming from a saved checkpoint using the `--checkpoint_path` flag:

**Standard training:**
```bash
python train_cifar10.py --model=otcfm --checkpoint_path=./results/otcfm/otcfm_cifar10_weights_step_20000.pt
```

**Locality training:**
```bash
python train_cifar10_locality.py --model=otcfm --lambda_local=1.0 --lambda_name=1e0 --checkpoint_path=./results/otcfm_local_1e0/otcfm_cifar10_local_1e0_weights_step_20000.pt
```

**Anchor training:**
```bash
python train_cifar10_anchor.py --model=otcfm --lambda_anchor=1.0 --lambda_name=1e0 --anchor_loss_type=full --checkpoint_path=./results/otcfm_anchor_full/otcfm_cifar10_anchor_1e0_weights_step_20000.pt
```

When resuming from a checkpoint, the training will automatically:
- Load model weights (both main model and EMA model)
- Restore optimizer and learning rate scheduler states  
- Continue training from the saved step (step counter is preserved)
- Load class centroids (for anchor training only)

## Multi-GPU Training

You can train all methods in parallel using multiple GPUs with PyTorch DistributedDataParallel:

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=NUM_GPUS train_cifar10_ddp.py --model=otcfm --lr=2e-4 --ema_decay=0.9999 --batch_size=128 --total_steps=400001 --save_step=20000 --parallel=True --master_addr=MASTER_ADDR --master_port=MASTER_PORT
```

**Note:** Parallel training may perform slightly worse than single GPU training due to batch normalization statistics computation in DataParallel.

## Evaluation

To compute the FID score from a trained model:

```bash
python compute_fid.py --model=otcfm --step=400000 --integration_method=dopri5
```

For other models, change `--model=otcfm` to `--model=icfm` or `--model=fm`.

## Pre-trained Weights

For easy reproducibility, you can download pre-trained model weights at 400,000 iterations:

- [OT-CFM weights](https://github.com/atong01/conditional-flow-matching/releases/download/1.0.4/otcfm_cifar10_weights_step_400000.pt) (FID: 3.5)
- [I-CFM weights](https://github.com/atong01/conditional-flow-matching/releases/download/1.0.4/cfm_cifar10_weights_step_400000.pt)
- [FM weights](https://github.com/atong01/conditional-flow-matching/releases/download/1.0.4/fm_cifar10_weights_step_400000.pt)

## Implementation Details

### Class-Conditioned Sampling
Both standard training and locality-regularized training use a `ClassConditionedSampler` that ensures all samples in a batch come from the same class. This is important for:
- Computing meaningful local loss metrics (samples should be semantically similar)
- Enforcing locality properties during training (for locality-regularized model)

The `--num_subclasses` parameter (default: 25) controls how finely each class is divided into subclasses for sampling, enabling finer-grained control over batch composition.

### Cluster Visualization
When `--visualize=True` (default), the training scripts will generate visualizations of the clustering results at startup. For each cluster created by K-means:
- A grid of sample images is saved showing what samples were grouped together
- Images are saved to the `./group/` directory
- Filenames follow the pattern: `class_XXX_orig_Y_sub_ZZ.png`
  - `XXX`: new cluster ID (0 to num_classes*num_subclasses-1)
  - `Y`: original CIFAR-10 class (0-9: airplane, car, bird, cat, deer, dog, frog, horse, ship, truck)
  - `ZZ`: subcluster number within the original class
- A `README.txt` summary file is created with cluster statistics

This helps verify that K-means successfully groups semantically similar images together.

### Local Loss Computation
The local loss measures the locality property of the learned vector field:

**Local Loss = ||v(u_t, t) - v(s_t, t) + s - u||²**

where u and s are samples from the same class, and u_t and s_t are interpolated with the same noise and time.

- In **standard training**: Computed for logging only (monitors implicit locality learning)
- In **locality training**: Used as an explicit regularization term with weight λ

### Training Parameters

**Common Parameters:**
- `--model`: Flow matching variant ("otcfm", "icfm", "fm", "si")
- `--lr`: Learning rate (default: 2e-4)
- `--batch_size`: Batch size (default: 128)
- `--total_steps`: Total training steps (default: 400001)
- `--save_step`: Checkpoint frequency (default: 20000)
- `--grad_clip`: Gradient clipping norm (default: 1.0)
- `--ema_decay`: EMA decay rate for model averaging (default: 0.9999)
- `--num_channel`: Base channel count for UNet (default: 128)
- `--num_subclasses`: Number of subclasses for class-conditioned sampling (default: 25)
- `--visualize`: Enable cluster visualization at startup (default: True)
- `--checkpoint_path`: Path to checkpoint for resuming training

