"""
TensorFlow/TPU training script for Locality-Regularized Conditional Flow Matching on CIFAR-10.

This script trains with explicit local regularization loss to enforce locality properties.
Samples batches from the same class and adds a regularization term: λ · ||v(u_t, t) - v(s_t, t) + s - u||²

Supports:
- Single GPU training
- Multi-GPU training with MirroredStrategy
- TPU training with TPUStrategy

Authors: Kilian Fatras
         Alexander Tong
         (TensorFlow port)

Usage:
    # Local GPU training
    python train_cifar10_locality.py --model=otcfm --batch_size=128 --lambda_local=1.0 --total_steps=400001
    
    # TPU training (on Google Cloud)
    python train_cifar10_locality.py --model=otcfm --batch_size=128 --use_tpu=True --tpu_name=your-tpu-name
"""

import os
import sys
import time
import argparse
import numpy as np
import tensorflow as tf

# Local imports
from unet import create_unet_cifar10
from conditional_flow_matching import get_flow_matcher
from utils import (
    load_cifar10, ClassConditionedSampler,
    ExponentialMovingAverage, WarmupSchedule,
    generate_and_save_samples, compute_local_loss,
    save_checkpoint, load_checkpoint, ema_update, visualize_clusters
)


def parse_args():
    parser = argparse.ArgumentParser(description='Train Locality-CFM on CIFAR-10 with TensorFlow/TPU')
    
    # Model
    parser.add_argument('--model', type=str, default='otcfm',
                       choices=['otcfm', 'icfm', 'fm', 'si'],
                       help='Flow matching model type')
    parser.add_argument('--num_channel', type=int, default=128,
                       help='Base channel of UNet')
    
    # Training
    parser.add_argument('--lr', type=float, default=2e-4,
                       help='Learning rate')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                       help='Gradient norm clipping')
    parser.add_argument('--total_steps', type=int, default=400001,
                       help='Total training steps')
    parser.add_argument('--warmup', type=int, default=5000,
                       help='Learning rate warmup steps')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size per replica')
    parser.add_argument('--ema_decay', type=float, default=0.9999,
                       help='EMA decay rate')
    
    # Locality regularization
    parser.add_argument('--lambda_local', type=float, default=1.0,
                       help='Weight for local regularization term')
    parser.add_argument('--lambda_name', type=str, default='1e0',
                       help='String identifier for experiment naming')
    parser.add_argument('--num_subclasses', type=int, default=25,
                       help='Number of subclasses for class-conditioned sampling')
    parser.add_argument('--visualize', type=bool, default=True,
                       help='Visualize clustering results by saving sample images')
    
    # TPU/GPU settings
    parser.add_argument('--use_tpu', type=bool, default=False,
                       help='Use TPU for training')
    parser.add_argument('--tpu_name', type=str, default=None,
                       help='TPU name (for Cloud TPU)')
    parser.add_argument('--tpu_zone', type=str, default=None,
                       help='TPU zone')
    parser.add_argument('--gcp_project', type=str, default=None,
                       help='GCP project name')
    parser.add_argument('--mixed_precision', type=bool, default=False,
                       help='Use mixed precision (bfloat16)')
    
    # Checkpointing
    parser.add_argument('--output_dir', type=str, default='./results/',
                       help='Output directory')
    parser.add_argument('--save_step', type=int, default=20000,
                       help='Checkpoint save frequency')
    parser.add_argument('--checkpoint_path', type=str, default='',
                       help='Path to checkpoint to resume from')
    
    # Logging
    parser.add_argument('--use_wandb', type=bool, default=False,
                       help='Use Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='cfm-tensorflow',
                       help='W&B project name')
    
    return parser.parse_args()


def setup_strategy(args):
    """Setup the distribution strategy based on available hardware."""
    if args.use_tpu:
        if args.tpu_name:
            resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
                tpu=args.tpu_name,
                zone=args.tpu_zone,
                project=args.gcp_project
            )
        else:
            resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
        
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.TPUStrategy(resolver)
        print(f"Running on TPU with {strategy.num_replicas_in_sync} replicas")
    else:
        gpus = tf.config.list_physical_devices('GPU')
        if len(gpus) > 1:
            strategy = tf.distribute.MirroredStrategy()
            print(f"Running on {len(gpus)} GPUs")
        elif len(gpus) == 1:
            strategy = tf.distribute.get_strategy()
            print("Running on single GPU")
        else:
            strategy = tf.distribute.get_strategy()
            print("Running on CPU")
    
    return strategy


def create_model(args, strategy):
    """Create and return the model within the strategy scope."""
    with strategy.scope():
        model = create_unet_cifar10(
            num_channels=args.num_channel,
            dropout=0.1
        )
        ema_model = create_unet_cifar10(
            num_channels=args.num_channel,
            dropout=0.1
        )
        
        # Build models with dummy input
        dummy_t = tf.zeros([1])
        dummy_x = tf.zeros([1, 32, 32, 3])
        model(dummy_t, dummy_x)
        ema_model(dummy_t, dummy_x)
        
        # Copy weights to EMA model
        for ema_var, model_var in zip(ema_model.trainable_variables, 
                                       model.trainable_variables):
            ema_var.assign(model_var)
        
        # Optimizer with warmup
        lr_schedule = WarmupSchedule(args.lr, args.warmup)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        
    return model, ema_model, optimizer


def train_step_locality(model, optimizer, x1, x0, FM, grad_clip, lambda_local, sigma=0.0):
    """
    Single training step with locality regularization.
    
    All samples in x1 should be from the same class for meaningful local loss.
    
    Parameters
    ----------
    model : keras.Model
        The UNet model
    optimizer : keras.optimizers.Optimizer
        The optimizer
    x1 : tf.Tensor
        Target images (from same class)
    x0 : tf.Tensor
        Source samples (noise)
    FM : ConditionalFlowMatcher
        Flow matcher instance
    grad_clip : float
        Gradient clipping value
    lambda_local : float
        Weight for local regularization
    sigma : float
        Noise level for flow matching
    
    Returns
    -------
    loss : tf.Tensor
        Total loss
    loss_fm : tf.Tensor
        Flow matching loss
    loss_local : tf.Tensor
        Local regularization loss
    grad_norm : tf.Tensor
        Gradient norm before clipping
    """
    with tf.GradientTape() as tape:
        # Sample location and conditional flow
        t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
        
        # Get model prediction
        vt = model(t, xt, training=True)
        
        # Standard flow matching loss
        loss_fm = tf.reduce_mean((vt - ut) ** 2)
        
        # Local regularization loss
        loss_local = compute_local_loss(model, x1, x0, sigma=sigma)
        
        # Combined loss
        loss = loss_fm + lambda_local * loss_local
    
    # Compute gradients
    gradients = tape.gradient(loss, model.trainable_variables)
    
    # Compute gradient norm before clipping
    grad_norm = tf.sqrt(sum([tf.reduce_sum(g ** 2) for g in gradients if g is not None]))
    
    # Clip gradients
    gradients, _ = tf.clip_by_global_norm(gradients, grad_clip)
    
    # Apply gradients
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return loss, loss_fm, loss_local, grad_norm


def train(args):
    """Main training function."""
    
    # Setup logging
    if args.use_wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=f"{args.model}_cifar10_local_{args.lambda_name}_tf",
            config=vars(args)
        )
    
    # Setup strategy
    strategy = setup_strategy(args)
    
    # Enable mixed precision if requested
    if args.mixed_precision:
        if args.use_tpu:
            tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')
        else:
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
        print("Mixed precision enabled")
    
    # Load data
    print("Loading CIFAR-10 dataset...")
    (x_train, y_train), (x_test, y_test) = load_cifar10()
    print(f"Training samples: {len(x_train)}, Test samples: {len(x_test)}")
    
    # Calculate global batch size
    global_batch_size = args.batch_size * strategy.num_replicas_in_sync
    print(f"Global batch size: {global_batch_size}")
    
    # Create class-conditioned sampler for locality training
    # This ensures all samples in a batch come from the same class
    class_sampler = ClassConditionedSampler(x_train, y_train, global_batch_size, num_classes=10, num_subclasses=args.num_subclasses)
    
    # Visualize clusters if requested
    if args.visualize:
        print("Visualizing class clusters...")
        visualize_clusters(class_sampler)
    
    # Create model
    print("Creating model...")
    model, ema_model, optimizer = create_model(args, strategy)
    
    # Print model info
    model_params = sum([tf.reduce_prod(v.shape).numpy() for v in model.trainable_variables])
    print(f"Model parameters: {model_params / 1e6:.2f}M")
    
    # Load checkpoint if provided
    start_step = 0
    save_dir = os.path.join(args.output_dir, f"{args.model}_local_{args.lambda_name}")
    if args.checkpoint_path:
        start_step = load_checkpoint(model, ema_model, optimizer, 
                                     args.checkpoint_path, f"{args.model}_local_{args.lambda_name}")
    
    # Get flow matcher
    FM = get_flow_matcher(args.model, sigma=0.0)
    
    # Training loop
    print(f"Starting locality training from step {start_step}...")
    print(f"Lambda (local regularization weight): {args.lambda_local}")
    os.makedirs(save_dir, exist_ok=True)
    
    step = start_step
    start_time = time.time()
    
    while step < args.total_steps:
        # Sample a batch from the same class
        x1, sampled_class = class_sampler.sample_batch_from_class()
        x0 = tf.random.normal(tf.shape(x1))
        
        # Training step with locality regularization
        loss, loss_fm, loss_local, grad_norm = train_step_locality(
            model, optimizer, x1, x0, FM, 
            args.grad_clip, args.lambda_local
        )
        
        # EMA update
        ema_update(model, ema_model, args.ema_decay)
        
        # Logging
        if step % 100 == 0:
            elapsed = time.time() - start_time
            steps_per_sec = (step - start_step + 1) / elapsed if elapsed > 0 else 0
            
            print(f"Step {step}/{args.total_steps} | "
                  f"Loss: {loss.numpy():.4f} | "
                  f"FM Loss: {loss_fm.numpy():.4f} | "
                  f"Local Loss: {loss_local.numpy():.4f} | "
                  f"Class: {sampled_class} | "
                  f"Grad Norm: {grad_norm.numpy():.4f} | "
                  f"Speed: {steps_per_sec:.2f} steps/s")
            
            if args.use_wandb:
                wandb.log({
                    'loss': loss.numpy(),
                    'loss_fm': loss_fm.numpy(),
                    'loss_local': loss_local.numpy(),
                    'grad_norm': grad_norm.numpy(),
                    'sampled_class': sampled_class,
                    'learning_rate': optimizer.learning_rate(step).numpy(),
                    'step': step
                }, step=step)
        
        # Save checkpoint and generate samples
        if args.save_step > 0 and step % args.save_step == 0 and step > 0:
            print(f"\nSaving checkpoint at step {step}...")
            
            # Generate and save samples
            generate_and_save_samples(model, save_dir, step, 
                                     num_samples=64, prefix="normal")
            generate_and_save_samples(ema_model, save_dir, step,
                                     num_samples=64, prefix="ema")
            
            # Save checkpoint
            save_checkpoint(model, ema_model, optimizer, step, save_dir, 
                          f"{args.model}_local_{args.lambda_name}")
            
            print(f"Checkpoint saved at step {step}\n")
        
        step += 1
    
    # Final save
    print("Training complete! Saving final checkpoint...")
    save_checkpoint(model, ema_model, optimizer, step, save_dir, 
                   f"{args.model}_local_{args.lambda_name}")
    generate_and_save_samples(ema_model, save_dir, step, num_samples=64, prefix="final")
    
    if args.use_wandb:
        wandb.finish()
    
    print("Done!")


if __name__ == "__main__":
    args = parse_args()
    train(args)
