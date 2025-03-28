#!/bin/bash
# prepare_usc_data.sh - Main script for USC data preparation
# Place this in egs/usc/

# Set paths
KALDI_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
USC_DIR=$KALDI_ROOT/egs/usc

echo "Starting USC data preparation..."

# Run the Python data preparation script from the local directory
python3 $USC_DIR/local/prepare_data.py

# Sort the created files
for subset in train dev test; do
    echo "Sorting files for $subset..."
    
    # Sort uttids (if they exist)
    if [ -f $USC_DIR/data/$subset/uttids ]; then
        sort $USC_DIR/data/$subset/uttids > $USC_DIR/data/$subset/uttids.sorted
        mv $USC_DIR/data/$subset/uttids.sorted $USC_DIR/data/$subset/uttids
    else
        echo "Warning: $USC_DIR/data/$subset/uttids not found"
    fi
    
    # Sort utt2spk (if they exist)
    if [ -f $USC_DIR/data/$subset/utt2spk ]; then
        sort $USC_DIR/data/$subset/utt2spk > $USC_DIR/data/$subset/utt2spk.sorted
        mv $USC_DIR/data/$subset/utt2spk.sorted $USC_DIR/data/$subset/utt2spk
    else
        echo "Warning: $USC_DIR/data/$subset/utt2spk not found"
    fi
    
    # Sort wav.scp (if they exist)
    if [ -f $USC_DIR/data/$subset/wav.scp ]; then
        sort $USC_DIR/data/$subset/wav.scp > $USC_DIR/data/$subset/wav.scp.sorted
        mv $USC_DIR/data/$subset/wav.scp.sorted $USC_DIR/data/$subset/wav.scp
    else
        echo "Warning: $USC_DIR/data/$subset/wav.scp not found"
    fi
    
    # Sort text (if they exist)
    if [ -f $USC_DIR/data/$subset/text ]; then
        sort $USC_DIR/data/$subset/text > $USC_DIR/data/$subset/text.sorted
        mv $USC_DIR/data/$subset/text.sorted $USC_DIR/data/$subset/text
    else
        echo "Warning: $USC_DIR/data/$subset/text not found"
    fi
    
    # Create spk2utt
    echo "Creating spk2utt for $subset..."
    if [ -f $USC_DIR/data/$subset/utt2spk ]; then
        $USC_DIR/utils/utt2spk_to_spk2utt.pl $USC_DIR/data/$subset/utt2spk > $USC_DIR/data/$subset/spk2utt
    else
        echo "Warning: Cannot create spk2utt; $USC_DIR/data/$subset/utt2spk not found"
    fi
done

echo "USC data preparation completed successfully!"