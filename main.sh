#!/bin/bash

# Path variables
KALDI_ROOT=~
USC_DATA=$KALDI_ROOT/egs/usc  # Directory containing wav, filesets, etc.
EXP=$KALDI_ROOT/egs/usc

# Create local directory for custom scripts
mkdir -p $EXP/local

# Copy all scripts to the appropriate locations
cp prepare_data.py $EXP/local/
cp prepare_all_data.sh $EXP/local/
cp extract_features.sh $EXP/local/
cp create_language_model.sh $EXP/local/
cp train_acoustic_model.sh $EXP/local/
cp decode_and_evaluate.sh $EXP/local/

# Make scripts executable
chmod +x $EXP/local/*.sh $EXP/local/*.py

# Step 1: Prepare data
echo "Step 1: Preparing data..."
$EXP/local/prepare_all_data.sh

# Step 2: Extract features
echo "Step 2: Extracting features..."
$EXP/local/extract_features.sh

# Step 3: Create language model
echo "Step 3: Creating language model..."
$EXP/local/create_language_model.sh

# Step 4: Train acoustic model
echo "Step 4: Training acoustic model..."
$EXP/local/train_acoustic_model.sh

# Step 5: Decode and evaluate
echo "Step 5: Decoding and evaluation..."
$EXP/local/decode_and_evaluate.sh

echo "All steps completed successfully!"