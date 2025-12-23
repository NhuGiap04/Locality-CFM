"""
TensorFlow FID computation script for Conditional Flow Matching on CIFAR-10.

This script computes the Fréchet Inception Distance (FID) for generated samples.
Uses tensorflow_gan for FID computation which is more memory efficient than cleanfid.

Authors: Kilian Fatras
         Alexander Tong
         (TensorFlow port)

Usage:
    python compute_fid.py --model=otcfm --step=400000 --num_gen=50000
    
    # With TPU for generation (FID uses CPU/GPU)
    python compute_fid.py --model=otcfm --step=400000 --use_tpu=True --tpu_name=your-tpu
"""

import os
import sys
import argparse
import numpy as np
import tensorflow as tf

# Local imports
from unet import create_unet_cifar10
from utils import load_cifar10, euler_integrate, rk4_integrate


def parse_args():
    parser = argparse.ArgumentParser(description='Compute FID for CIFAR-10 CFM models')
    
    # Model
    parser.add_argument('--model', type=str, default='otcfm',
                       help='Flow matching model type')
    parser.add_argument('--num_channel', type=int, default=128,
                       help='Base channel of UNet')
    
    # FID computation
    parser.add_argument('--num_gen', type=int, default=50000,
                       help='Number of samples to generate for FID')
    parser.add_argument('--batch_size_fid', type=int, default=256,
                       help='Batch size for FID computation (smaller = less memory)')
    parser.add_argument('--integration_steps', type=int, default=100,
                       help='Number of ODE integration steps')
    parser.add_argument('--integration_method', type=str, default='euler',
                       choices=['euler', 'rk4'],
                       help='ODE integration method')
    
    # Checkpoint
    parser.add_argument('--input_dir', type=str, default='./results',
                       help='Directory containing checkpoints')
    parser.add_argument('--step', type=int, default=400000,
                       help='Training step to evaluate')
    
    # TPU/GPU settings
    parser.add_argument('--use_tpu', type=bool, default=False,
                       help='Use TPU for generation')
    parser.add_argument('--tpu_name', type=str, default=None,
                       help='TPU name')
    
    return parser.parse_args()


def load_inception_model():
    """
    Load InceptionV3 model for FID computation.
    Uses the pool3 layer output (2048-dim features).
    
    Returns
    -------
    keras.Model
        InceptionV3 model with pool3 output
    """
    # Load InceptionV3 with ImageNet weights
    base_model = tf.keras.applications.InceptionV3(
        include_top=False,
        weights='imagenet',
        input_shape=(299, 299, 3),
        pooling='avg'
    )
    
    # The output is already 2048-dim from global average pooling
    return base_model


def preprocess_for_inception(images):
    """
    Preprocess images for InceptionV3.
    
    Parameters
    ----------
    images : tf.Tensor
        Images in [-1, 1] range, shape [N, 32, 32, 3]
    
    Returns
    -------
    tf.Tensor
        Preprocessed images, shape [N, 299, 299, 3]
    """
    # Rescale from [-1, 1] to [0, 255]
    images = (images + 1.0) * 127.5
    images = tf.clip_by_value(images, 0, 255)
    
    # Resize to 299x299
    images = tf.image.resize(images, [299, 299], method='bilinear')
    
    # Preprocess for InceptionV3 (expects [-1, 1] range)
    images = tf.keras.applications.inception_v3.preprocess_input(images)
    
    return images


def compute_inception_features(images, inception_model, batch_size=256):
    """
    Compute InceptionV3 features for a batch of images.
    
    Parameters
    ----------
    images : np.ndarray
        Images in [-1, 1] range, shape [N, 32, 32, 3]
    inception_model : keras.Model
        InceptionV3 model
    batch_size : int
        Batch size for processing
    
    Returns
    -------
    np.ndarray
        Features, shape [N, 2048]
    """
    num_images = len(images)
    features = []
    
    for i in range(0, num_images, batch_size):
        batch = images[i:i + batch_size]
        batch = preprocess_for_inception(batch)
        feat = inception_model(batch, training=False)
        features.append(feat.numpy())
        
        if (i // batch_size) % 10 == 0:
            print(f"  Processed {min(i + batch_size, num_images)}/{num_images} images")
    
    return np.concatenate(features, axis=0)


def compute_fid_from_features(real_features, gen_features):
    """
    Compute FID from pre-computed features.
    
    Parameters
    ----------
    real_features : np.ndarray
        Features from real images, shape [N, 2048]
    gen_features : np.ndarray
        Features from generated images, shape [M, 2048]
    
    Returns
    -------
    float
        FID score
    """
    # Compute statistics
    mu_real = np.mean(real_features, axis=0)
    mu_gen = np.mean(gen_features, axis=0)
    
    sigma_real = np.cov(real_features, rowvar=False)
    sigma_gen = np.cov(gen_features, rowvar=False)
    
    # Compute FID
    diff = mu_real - mu_gen
    
    # Product of covariances
    from scipy import linalg
    covmean, _ = linalg.sqrtm(sigma_real @ sigma_gen, disp=False)
    
    # Numerical stability
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = diff @ diff + np.trace(sigma_real) + np.trace(sigma_gen) - 2 * np.trace(covmean)
    
    return fid


def generate_samples(model, num_samples, batch_size, num_steps=100, method='euler'):
    """
    Generate samples using the trained model.
    
    Parameters
    ----------
    model : keras.Model
        The UNet model
    num_samples : int
        Total number of samples to generate
    batch_size : int
        Batch size for generation
    num_steps : int
        Number of ODE integration steps
    method : str
        Integration method ('euler' or 'rk4')
    
    Returns
    -------
    np.ndarray
        Generated samples, shape [num_samples, 32, 32, 3]
    """
    samples = []
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    print(f"Generating {num_samples} samples in {num_batches} batches...")
    
    for i in range(num_batches):
        current_batch_size = min(batch_size, num_samples - i * batch_size)
        
        # Start from noise
        x0 = tf.random.normal([current_batch_size, 32, 32, 3])
        
        # Integrate ODE
        if method == 'euler':
            x1 = euler_integrate(model, x0, num_steps)
        else:
            x1 = rk4_integrate(model, x0, num_steps)
        
        # Clip to valid range
        x1 = tf.clip_by_value(x1, -1.0, 1.0)
        samples.append(x1.numpy())
        
        if (i + 1) % 10 == 0:
            print(f"  Generated batch {i + 1}/{num_batches}")
    
    return np.concatenate(samples, axis=0)[:num_samples]


def main():
    args = parse_args()
    
    # Setup GPU memory growth
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    
    print("=" * 60)
    print("FID Computation for CIFAR-10 CFM")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Step: {args.step}")
    print(f"Num generated samples: {args.num_gen}")
    print(f"Integration method: {args.integration_method}")
    print(f"Integration steps: {args.integration_steps}")
    print("=" * 60)
    
    # Load model
    print("\nLoading model...")
    model = create_unet_cifar10(num_channels=args.num_channel, dropout=0.0)
    
    # Build model
    dummy_t = tf.zeros([1])
    dummy_x = tf.zeros([1, 32, 32, 3])
    model(dummy_t, dummy_x)
    
    # Load weights
    checkpoint_path = os.path.join(
        args.input_dir, args.model, 
        f"{args.model}_ema_step_{args.step}.h5"
    )
    
    if not os.path.exists(checkpoint_path):
        # Try alternative naming
        checkpoint_path = os.path.join(
            args.input_dir, args.model,
            f"{args.model}_cifar10_weights_step_{args.step}.h5"
        )
    
    if os.path.exists(checkpoint_path):
        model.load_weights(checkpoint_path)
        print(f"Loaded weights from {checkpoint_path}")
    else:
        print(f"ERROR: Could not find checkpoint at {checkpoint_path}")
        print("Please ensure the checkpoint file exists.")
        sys.exit(1)
    
    # Generate samples
    print("\nGenerating samples...")
    gen_samples = generate_samples(
        model, args.num_gen, args.batch_size_fid,
        num_steps=args.integration_steps,
        method=args.integration_method
    )
    print(f"Generated {len(gen_samples)} samples")
    
    # Load real CIFAR-10 data
    print("\nLoading CIFAR-10 training data...")
    (x_train, _), _ = load_cifar10()
    print(f"Loaded {len(x_train)} real images")
    
    # Load Inception model
    print("\nLoading InceptionV3 model...")
    inception_model = load_inception_model()
    
    # Compute features for generated samples
    print("\nComputing features for generated samples...")
    gen_features = compute_inception_features(gen_samples, inception_model, args.batch_size_fid)
    print(f"Generated features shape: {gen_features.shape}")
    
    # Compute features for real samples
    print("\nComputing features for real samples...")
    # Use the same number of real samples as generated
    num_real = min(len(x_train), args.num_gen)
    indices = np.random.choice(len(x_train), num_real, replace=False)
    real_samples = x_train[indices]
    real_features = compute_inception_features(real_samples, inception_model, args.batch_size_fid)
    print(f"Real features shape: {real_features.shape}")
    
    # Compute FID
    print("\nComputing FID score...")
    fid_score = compute_fid_from_features(real_features, gen_features)
    
    print("\n" + "=" * 60)
    print(f"FID Score: {fid_score:.4f}")
    print("=" * 60)
    
    # Save results
    results_file = os.path.join(args.input_dir, args.model, f"fid_step_{args.step}.txt")
    with open(results_file, 'w') as f:
        f.write(f"Model: {args.model}\n")
        f.write(f"Step: {args.step}\n")
        f.write(f"Num generated: {args.num_gen}\n")
        f.write(f"Integration method: {args.integration_method}\n")
        f.write(f"Integration steps: {args.integration_steps}\n")
        f.write(f"FID: {fid_score:.4f}\n")
    print(f"Results saved to {results_file}")


if __name__ == "__main__":
    main()
