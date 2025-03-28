#!/bin/bash
# extract_cmvn.sh - Extract CMVN statistics for DNN training
# Place this in egs/usc/

# Set paths
KALDI_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
USC_DIR=$KALDI_ROOT/egs/usc

# Source path.sh to set up the environment
source $USC_DIR/path.sh
source $USC_DIR/cmd.sh

echo "Extracting CMVN statistics for DNN training..."

# Check if feature files exist
for dataset in train dev test; do
    if [ ! -f $USC_DIR/data/$dataset/feats.scp ]; then
        echo "Error: Feature file feats.scp for $dataset not found."
        echo "Please extract MFCC features first."
        exit 1
    fi
done

# Extract CMVN statistics for each dataset
for dataset in train dev test; do
    echo "Computing CMVN statistics for $dataset..."
    
    # Compute per-speaker CMVN stats
    echo "Computing per-speaker CMVN stats..."
    compute-cmvn-stats --spk2utt=ark:$USC_DIR/data/$dataset/spk2utt \
        scp:$USC_DIR/data/$dataset/feats.scp \
        ark:$USC_DIR/data/$dataset/${dataset}_cmvn_speaker.ark || {
        echo "Error: Computing per-speaker CMVN stats for $dataset failed."
        exit 1
    }
    
    # Compute per-utterance CMVN stats
    echo "Computing per-utterance CMVN stats..."
    compute-cmvn-stats scp:$USC_DIR/data/$dataset/feats.scp \
        ark:$USC_DIR/data/$dataset/${dataset}_cmvn_snt.ark || {
        echo "Error: Computing per-utterance CMVN stats for $dataset failed."
        exit 1
    }
    
    echo "CMVN statistics for $dataset computed successfully."
done

echo "CMVN statistics extraction completed successfully."