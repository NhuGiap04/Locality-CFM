# CIFAR-10 Experiments using TensorFlow/TPU

This directory contains the TensorFlow implementation of Conditional Flow Matching for CIFAR-10, optimized for Google Cloud TPU.

## Features

- **TPU Support**: Native TPU training using `tf.distribute.TPUStrategy`
- **Multi-GPU Support**: Automatic multi-GPU training with `MirroredStrategy`
- **Mixed Precision**: BFloat16 support for TPU, Float16 for GPU
- **Memory Efficient FID**: Uses TensorFlow's InceptionV3 directly, much lower memory usage than cleanfid

## Requirements

```bash
pip install tensorflow>=2.12.0
pip install tensorflow-probability  # For advanced OT
pip install POT  # For exact OT computation
pip install scipy  # For FID computation
pip install wandb  # Optional, for logging
```

For TPU training on Google Cloud:
```bash
pip install cloud-tpu-client
```

## Directory Structure

```
cifar_tensorflow/
├── __init__.py
├── unet.py                      # UNet model implementation
├── conditional_flow_matching.py  # Flow matching algorithms
├── optimal_transport.py          # OT plan samplers
├── utils.py                      # Training utilities
├── train_cifar10.py              # Standard training script
├── train_cifar10_locality.py     # Locality-regularized training
├── compute_fid.py                # FID computation
└── README.md
```

## Training

### Local GPU Training

**Standard OT-CFM:**
```bash
python train_cifar10.py --model=otcfm --lr=2e-4 --ema_decay=0.9999 --batch_size=128 --total_steps=400001 --num_subclasses=25 --visualize=True
```

**Independent CFM (I-CFM):**
```bash
python train_cifar10.py --model=icfm --lr=2e-4 --batch_size=128 --total_steps=400001 --visualize=True
```

**Original Flow Matching (FM):**
```bash
python train_cifar10.py --model=fm --lr=2e-4 --batch_size=128 --total_steps=400001 --visualize=True
```

**Variance Preserving (Stochastic Interpolant):**
```bash
python train_cifar10.py --model=si --lr=2e-4 --batch_size=128 --total_steps=400001 --visualize=True
```

### TPU Training (Google Cloud)

**On Cloud TPU VM:**
```bash
python train_cifar10.py \
    --model=otcfm \
    --batch_size=128 \
    --total_steps=400001 \
    --use_tpu=True
```

**On Colab/Kaggle with TPU:**
```python
# The script auto-detects TPU
!python train_cifar10.py --model=otcfm --batch_size=128 --use_tpu=True
```

**With specific TPU name:**
```bash
python train_cifar10.py \
    --model=otcfm \
    --batch_size=128 \
    --use_tpu=True \
    --tpu_name=your-tpu-name \
    --tpu_zone=us-central1-b \
    --gcp_project=your-project
```

### Locality-Regularized Training

Trains with explicit local regularization to enforce locality properties:

```bash
python train_cifar10_locality.py \
    --model=otcfm \
    --batch_size=128 \
    --lambda_local=1.0 \
    --lambda_name=1e0 \
    --num_subclasses=25 \
    --visualize=True \
    --total_steps=400001
```

**Key Parameters:**
- `--lambda_local`: Weight for local regularization term (default: 1.0)
- `--lambda_name`: String identifier for experiment naming (e.g., "1e0", "1e-1")
- `--num_subclasses`: Number of subclasses for class-conditioned sampling (default: 25)
- `--visualize`: Visualize clustering results by saving sample images from each cluster (default: True)

### Training with Mixed Precision

For faster training and lower memory usage:

```bash
python train_cifar10.py --model=otcfm --batch_size=256 --mixed_precision=True
```

### Training with Weights & Biases

```bash
python train_cifar10.py --model=otcfm --use_wandb=True --wandb_project=my-cfm-project
```

## FID Evaluation

Compute FID score for a trained model:

```bash
python compute_fid.py \
    --model=otcfm \
    --step=400000 \
    --num_gen=50000 \
    --batch_size_fid=256
```

**Memory-efficient FID (for limited GPU memory):**
```bash
python compute_fid.py \
    --model=otcfm \
    --step=400000 \
    --num_gen=10000 \
    --batch_size_fid=64
```

## System Requirements

### Memory Requirements

| Configuration | GPU Memory | Recommended |
|--------------|------------|-------------|
| batch_size=32 | ~4 GB | Testing |
| batch_size=64 | ~6 GB | Small GPU |
| batch_size=128 | ~10 GB | Standard |
| batch_size=256 (mixed) | ~12 GB | Fast training |

### TPU Requirements

- TPU v2 or v3 recommended
- Cloud TPU VM or Colab Pro+
- 8-16 GB HBM per core

### FID Computation Memory

The TensorFlow implementation is **much more memory-efficient** than the PyTorch cleanfid version:

| Setting | Memory Usage |
|---------|-------------|
| batch_size_fid=256 | ~4 GB |
| batch_size_fid=128 | ~2.5 GB |
| batch_size_fid=64 | ~1.5 GB |

## Key Differences from PyTorch Version

1. **Data Format**: TensorFlow uses NHWC (batch, height, width, channels) instead of NCHW
2. **ODE Integration**: Uses custom Euler/RK4 integrators (no torchdiffeq equivalent)
3. **FID Computation**: Uses TensorFlow's InceptionV3 directly instead of cleanfid
4. **Optimal Transport**: Supports both CPU (POT library) and TPU-friendly (Sinkhorn) implementations

## Model Conversion

To convert PyTorch checkpoints to TensorFlow format, you would need to:
1. Load the PyTorch checkpoint
2. Map the weight names between frameworks
3. Transpose convolutional weights (PyTorch: OIHW, TensorFlow: HWIO)
4. Save in TensorFlow format

A conversion script is not included but can be implemented if needed.

## Troubleshooting

### TPU Connection Issues
```python
# Check TPU availability
import tensorflow as tf
try:
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
    print(f"TPU found: {resolver.cluster_spec()}")
except ValueError:
    print("No TPU found")
```

### Memory Issues
- Reduce `batch_size` or `batch_size_fid`
- Enable mixed precision with `--mixed_precision=True`
- For FID, use smaller `num_gen`

### NaN Loss
- Check if learning rate is too high
- Ensure data normalization is correct (should be [-1, 1])
- Try gradient clipping with smaller value

## References

- [Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport](https://arxiv.org/abs/2302.00482)
- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)
- [TensorFlow TPU Guide](https://cloud.google.com/tpu/docs/tutorials)
