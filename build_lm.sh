#!/bin/bash
# build_lm.sh - Build unigram and bigram language models using IRSTLM
# Place this in egs/usc/

# Set paths
KALDI_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
USC_DIR=$KALDI_ROOT/egs/usc

# Source path.sh to access the IRSTLM tools
source $USC_DIR/path.sh

echo "Building language models..."

# Ensure the directories exist
mkdir -p $USC_DIR/data/local/lm_tmp
mkdir -p $USC_DIR/data/local/dict

# Check if training text exists
if [ ! -f $USC_DIR/data/train/text ]; then
    echo "Error: text file not found in $USC_DIR/data/train/"
    echo "Please run data preparation scripts first."
    exit 1
fi

# Create proper lm_train.text without utterance IDs
echo "Creating proper language model training data..."

# Extract only the phoneme sequences from the text file (remove utterance IDs)
cat $USC_DIR/data/train/text | cut -d' ' -f2- > $USC_DIR/data/local/dict/phoneme_sequences.txt

# Create proper lm_train.text with <s> and </s> markers (no utterance IDs)
cat $USC_DIR/data/local/dict/phoneme_sequences.txt | \
    awk '{print "<s> " $0 " </s>"}' > $USC_DIR/data/local/dict/lm_train.text

echo "Created cleaned lm_train.text with sentence markers"

# Display sample of the LM training data
echo "Sample of LM training data (first 3 lines):"
head -n 3 $USC_DIR/data/local/dict/lm_train.text

# Build unigram language model
echo "Building unigram language model..."
build-lm.sh -i $USC_DIR/data/local/dict/lm_train.text -n 1 -o $USC_DIR/data/local/lm_tmp/lm_phone_ug.ilm.gz

# Check if the command succeeded
if [ $? -ne 0 ]; then
    echo "Error: Failed to build unigram language model."
    echo "Make sure IRSTLM is properly installed and available in your path."
    exit 1
fi

# Build bigram language model
echo "Building bigram language model..."
build-lm.sh -i $USC_DIR/data/local/dict/lm_train.text -n 2 -o $USC_DIR/data/local/lm_tmp/lm_phone_bg.ilm.gz

# Check if the command succeeded
if [ $? -ne 0 ]; then
    echo "Error: Failed to build bigram language model."
    exit 1
fi

echo "Language models built successfully."
echo "Unigram model: $USC_DIR/data/local/lm_tmp/lm_phone_ug.ilm.gz"
echo "Bigram model: $USC_DIR/data/local/lm_tmp/lm_phone_bg.ilm.gz"
echo "Next step: Run compile_lm.sh to convert to ARPA format"