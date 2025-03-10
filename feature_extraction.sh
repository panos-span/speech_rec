#!/bin/bash

# Path variables
KALDI_ROOT=~
EXP=$KALDI_ROOT/egs/usc

# Create directories for features and log files
mkdir -p $EXP/mfcc
mkdir -p $EXP/exp/make_mfcc
mkdir -p $EXP/exp/compute_cmvn_stats

# Extract MFCC features for each dataset
for dataset in train dev test; do
  # Extract MFCCs
  steps/make_mfcc.sh --nj 4 --cmd "run.pl" \
    $EXP/data/$dataset $EXP/exp/make_mfcc/$dataset $EXP/mfcc
    
  # Compute CMVN statistics
  steps/compute_cmvn_stats.sh \
    $EXP/data/$dataset $EXP/exp/compute_cmvn_stats/$dataset $EXP/mfcc
    
  # Validate the data directory with features
  utils/validate_data_dir.sh $EXP/data/$dataset
done

echo "Feature extraction completed"