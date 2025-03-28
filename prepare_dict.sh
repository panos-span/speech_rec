#!/bin/bash
# prepare_dict.sh - Script to prepare dictionary files
# Place this in egs/usc/

# Set paths
KALDI_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
USC_DIR=$KALDI_ROOT/egs/usc

echo "Starting dictionary preparation..."

# Run the Python dictionary preparation script
python3 $USC_DIR/local/prepare_dict.py

echo "Dictionary preparation completed successfully!"