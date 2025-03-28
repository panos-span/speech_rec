#!/bin/bash
# sort_files.sh - Script to sort data files and create spk2utt
# Place this in egs/usc/

# Set paths
KALDI_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
USC_DIR=$KALDI_ROOT/egs/usc

# Source path.sh to set up the environment
source $USC_DIR/path.sh

echo "Sorting data files and creating spk2utt..."

# Process each subset
for subset in train dev test; do
    echo "Processing $subset set..."
    
    # Check if the directory exists
    if [ ! -d $USC_DIR/data/$subset ]; then
        echo "Error: Directory $USC_DIR/data/$subset not found"
        continue
    fi
    
    # Sort wav.scp
    if [ -f $USC_DIR/data/$subset/wav.scp ]; then
        echo "Sorting wav.scp..."
        sort $USC_DIR/data/$subset/wav.scp > $USC_DIR/data/$subset/wav.scp.sorted
        mv $USC_DIR/data/$subset/wav.scp.sorted $USC_DIR/data/$subset/wav.scp
    else
        echo "Warning: wav.scp not found in $USC_DIR/data/$subset"
    fi
    
    # Sort text
    if [ -f $USC_DIR/data/$subset/text ]; then
        echo "Sorting text..."
        sort $USC_DIR/data/$subset/text > $USC_DIR/data/$subset/text.sorted
        mv $USC_DIR/data/$subset/text.sorted $USC_DIR/data/$subset/text
    else
        echo "Warning: text not found in $USC_DIR/data/$subset"
    fi
    
    # Sort utt2spk
    if [ -f $USC_DIR/data/$subset/utt2spk ]; then
        echo "Sorting utt2spk..."
        sort $USC_DIR/data/$subset/utt2spk > $USC_DIR/data/$subset/utt2spk.sorted
        mv $USC_DIR/data/$subset/utt2spk.sorted $USC_DIR/data/$subset/utt2spk
        
        # Create spk2utt from utt2spk
        echo "Creating spk2utt..."
        $USC_DIR/utils/utt2spk_to_spk2utt.pl $USC_DIR/data/$subset/utt2spk > $USC_DIR/data/$subset/spk2utt
    else
        echo "Warning: utt2spk not found in $USC_DIR/data/$subset"
    fi
    
    echo "Processing of $subset set completed."
done

echo "Sorting of data files and creation of spk2utt completed."