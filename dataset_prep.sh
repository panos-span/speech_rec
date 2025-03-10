#!/bin/bash

# Path variables - adjust these to your environment
KALDI_ROOT=~
USC_DATA=$KALDI_ROOT/egs/usc  # Directory containing wav, filesets, etc.
OUTPUT_DIR=$KALDI_ROOT/egs/usc/data

# Change to the Kaldi USC directory
cd $KALDI_ROOT/egs/usc

# Prepare data for each dataset
for dataset in train dev test; do
  python local/prepare_data.py \
    --data-dir $USC_DATA \
    --dataset $dataset \
    --output-dir $OUTPUT_DIR/$dataset
    
  # Verify the data directory
  utils/validate_data_dir.sh --no-feats $OUTPUT_DIR/$dataset
done

echo "All data preparation completed"