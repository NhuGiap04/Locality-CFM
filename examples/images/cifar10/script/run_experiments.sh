#!/bin/bash

# Script to run CIFAR-10 experiments with locality and anchor regularization
# Author: Generated for conditional-flow-matching experiments

set -e  # Exit on error

# Navigate to the cifar10 directory
cd "$(dirname "$0")"

echo "========================================"
echo "Running CIFAR-10 Training Experiments"
echo "========================================"
echo ""

# Configuration
MODEL="otcfm"
BATCH_SIZE=128
TOTAL_STEPS=400001
SAVE_STEP=20000
LR=2e-4
NUM_CHANNEL=128

# ========================================
# 1. Train with Locality Regularization
# ========================================
echo "Starting training with locality regularization..."
echo "Model: ${MODEL}"
echo "Lambda Local: 1.0"
echo "Local Lambda Size: medium"
echo ""

python train_cifar10_locality.py \
    --model="${MODEL}" \
    --batch_size=${BATCH_SIZE} \
    --total_steps=${TOTAL_STEPS} \
    --save_step=${SAVE_STEP} \
    --lr=${LR} \
    --num_channel=${NUM_CHANNEL} \
    --lambda_local=1.0 \
    --local_lambda_size="medium" \
    --output_dir="./results/"

echo ""
echo "✓ Locality regularization training completed"
echo ""

# ========================================
# 2. Train with Anchor Regularization (Full Loss)
# ========================================
echo "Starting training with anchor regularization (full loss)..."
echo "Model: ${MODEL}"
echo "Lambda Anchor: 1.0"
echo "Anchor Loss Type: full"
echo ""

python train_cifar10_anchor.py \
    --model="${MODEL}" \
    --batch_size=${BATCH_SIZE} \
    --total_steps=${TOTAL_STEPS} \
    --save_step=${SAVE_STEP} \
    --lr=${LR} \
    --num_channel=${NUM_CHANNEL} \
    --lambda_anchor=1.0 \
    --anchor_loss_type="full" \
    --centroid_update_freq=1000 \
    --output_dir="./results/"

echo ""
echo "✓ Anchor regularization (full) training completed"
echo ""

echo "========================================"
echo "All experiments completed!"
echo "========================================"
echo ""
echo "Results are saved in ./results/"
echo "Check wandb for training curves and FID scores"
