#!/bin/bash
# ============================================================
# ChatEvac ControlNet Training Script
# Finetunes Stable Diffusion v1.5 + ControlNet on floor plan
# datasets for building wall/exit segmentation.
#
# Usage:
#   1. Set the paths below to match your environment.
#   2. Run: bash train.sh
# ============================================================

# --- Configuration (edit these paths) ---
# HuggingFace model identifier or local path to SD 1.5
export MODEL_DIR="stable-diffusion-v1-5/stable-diffusion-v1-5"

# Directory where training checkpoints will be saved
# (relative to the script location, i.e. ChatEvac/checkpoint)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export OUTPUT_DIR="$SCRIPT_DIR/checkpoint"

# Path to the training dataset folder.
# Expected structure: {TRAIN_DATA_DIR}/source/  (input floor plans)
#                      {TRAIN_DATA_DIR}/target/  (ground-truth segmentations)
export TRAIN_DATA_DIR="$SCRIPT_DIR/dataset"

# A sample image from the dataset for validation during training
export VALIDATION_IMAGE="$SCRIPT_DIR/dataset/source/example.png"

# Validation prompt: annotate walls as white, indoor areas as black,
# outdoor doors as red lines — matching the 3-color floor plan convention.
export VALIDATION_PROMPT="Identify building walls in the image; annotate pixels corresponding to walls and outdoor areas as white, annotate indoor areas as black, ignore indoor doors, and represent doors leading outdoors with red lines."

# --- Training (adjust batch size / epochs / precision as needed) ---
accelerate launch train_controlnet.py \
  --pretrained_model_name_or_path="$MODEL_DIR" \
  --output_dir="$OUTPUT_DIR" \
  --train_data_dir="$TRAIN_DATA_DIR" \
  --resolution=512 \
  --learning_rate=1e-5 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --use_8bit_adam \
  --num_train_epochs=100 \
  --checkpointing_steps=10000 \
  --validation_image="$VALIDATION_IMAGE" \
  --validation_prompt="$VALIDATION_PROMPT" \
  --caption_column="prompt" \
  --image_column="target" \
  --conditioning_image_column="source" \
  --resume_from_checkpoint="latest"
