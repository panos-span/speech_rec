#!/bin/bash
# extract_alignments.sh - Extract triphone alignments for DNN training
# Place this in egs/usc/

# Set paths
KALDI_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
USC_DIR=$KALDI_ROOT/egs/usc

# Source path.sh to set up the environment
source $USC_DIR/path.sh
source $USC_DIR/cmd.sh

echo "Extracting triphone alignments for DNN training..."

# Check if triphone model exists
if [ ! -f $USC_DIR/exp/tri1/final.mdl ]; then
    echo "Error: Triphone model not found. Please train the triphone model first."
    exit 1
fi

# Function to align a dataset
align_dataset() {
    local dataset=$1
    
    echo "Aligning $dataset data..."
    
    steps/align_si.sh --nj 1 --cmd "$train_cmd" \
        $USC_DIR/data/$dataset $USC_DIR/data/lang $USC_DIR/exp/tri1 \
        $USC_DIR/exp/tri1_ali_$dataset || {
        echo "Error: Alignment of $dataset failed."
        return 1
    }
    
    echo "Alignment of $dataset completed successfully."
    return 0
}

# Align all datasets
align_dataset "train"
align_dataset "dev"
align_dataset "test"

echo "All alignments completed."

# Convert alignments to phone labels for DNN training
echo "Converting alignments to phone labels..."

for dataset in train dev test; do
    echo "Converting alignments for $dataset..."
    
    # Create output directory
    mkdir -p $USC_DIR/exp/tri1_ali_$dataset/pdf_ids
    
    # Convert alignments to pdf-ids
    ali-to-pdf $USC_DIR/exp/tri1/final.mdl \
        "ark:gunzip -c $USC_DIR/exp/tri1_ali_$dataset/ali.*.gz |" \
        "ark,t:$USC_DIR/exp/tri1_ali_$dataset/pdf_ids/pdf_ids.$dataset.txt" || {
        echo "Error: Converting alignments to pdf-ids for $dataset failed."
        exit 1
    }
    
    echo "Alignment conversion for $dataset completed."
done

echo "Alignment extraction and conversion completed successfully."