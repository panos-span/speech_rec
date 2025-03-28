#!/bin/bash
# create_hclg.sh - Create HCLG decoding graphs for unigram and bigram language models
# Place this in egs/usc/

# Set paths
KALDI_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
USC_DIR=$KALDI_ROOT/egs/usc

# Source path.sh to set up the environment
source $USC_DIR/path.sh
source $USC_DIR/cmd.sh

echo "Creating HCLG decoding graphs..."

# Check if monophone model exists
mono_dir=$USC_DIR/exp/mono
if [ ! -d $mono_dir ]; then
    echo "Error: Monophone model directory $mono_dir not found."
    echo "Please run steps/train_mono.sh first to train a monophone model."
    echo "Example command: steps/train_mono.sh --nj 4 data/train data/lang exp/mono"
    exit 1
fi

# Check if final.mdl exists in the monophone directory
if [ ! -f $mono_dir/final.mdl ]; then
    echo "Error: $mono_dir/final.mdl not found. Monophone training may not have completed successfully."
    exit 1
fi

# Check if lang directories with G.fst files exist
for lm_type in ug bg; do
    lang_dir=$USC_DIR/data/lang_$lm_type
    if [ ! -f $lang_dir/G.fst ]; then
        echo "Error: G.fst not found in $lang_dir"
        echo "Please run format_lm.sh first to create G.fst files."
        exit 1
    fi
done

# Create HCLG graph for unigram language model
echo "Creating HCLG graph for unigram language model..."
graph_dir=$mono_dir/graph_ug
utils/mkgraph.sh --mono $USC_DIR/data/lang_ug $mono_dir $graph_dir

# Check if HCLG.fst was created for unigram model
if [ ! -f $graph_dir/HCLG.fst ]; then
    echo "Error: Failed to create HCLG.fst for unigram language model."
    exit 1
fi

echo "Unigram HCLG graph created successfully at $graph_dir/HCLG.fst"

# Create HCLG graph for bigram language model
echo "Creating HCLG graph for bigram language model..."
graph_dir=$mono_dir/graph_bg
utils/mkgraph.sh --mono $USC_DIR/data/lang_bg $mono_dir $graph_dir

# Check if HCLG.fst was created for bigram model
if [ ! -f $graph_dir/HCLG.fst ]; then
    echo "Error: Failed to create HCLG.fst for bigram language model."
    exit 1
fi

echo "Bigram HCLG graph created successfully at $graph_dir/HCLG.fst"

echo "HCLG decoding graphs created successfully."
echo "You can now proceed with decoding using steps/decode.sh"