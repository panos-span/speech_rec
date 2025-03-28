#!/bin/bash
# train_mono.sh - Script to train a monophone GMM-HMM acoustic model
# Place this in egs/usc/

# Set paths
KALDI_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
USC_DIR=$KALDI_ROOT/egs/usc

# Source path.sh to set up the environment
source $USC_DIR/path.sh
source $USC_DIR/cmd.sh

echo "Training a monophone GMM-HMM acoustic model..."

# Check if MFCC features have been extracted
if [ ! -f $USC_DIR/data/train/feats.scp ]; then
    echo "Error: MFCC features have not been extracted. Please run feature extraction first."
    exit 1
fi

# Check if language model has been prepared
if [ ! -f $USC_DIR/data/lang/L.fst ]; then
    echo "Error: Language model has not been prepared. Please prepare the language model first."
    exit 1
fi

# Create output directory for monophone model
mkdir -p $USC_DIR/exp/mono

# Train monophone model
echo "Starting monophone training..."
steps/train_mono.sh --nj 1 --cmd "$train_cmd" \
    $USC_DIR/data/train $USC_DIR/data/lang $USC_DIR/exp/mono || {
    echo "Error: Monophone training failed."
    exit 1
}

echo "Monophone training completed successfully."
echo "Model saved in $USC_DIR/exp/mono"

# Show some statistics
echo "Final model statistics:"
gmm-info $USC_DIR/exp/mono/final.mdl

echo "Number of phones:"
am-info $USC_DIR/exp/mono/final.mdl | grep phones

echo "Number of states:"
am-info $USC_DIR/exp/mono/final.mdl | grep states

echo "Number of Gaussians:"
am-info $USC_DIR/exp/mono/final.mdl | grep 'total #Gaussians'

echo "Training complete."