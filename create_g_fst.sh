#!/bin/bash
# create_g_fst.sh - Script to create G.fst following TIMIT procedure
# Place this in egs/usc/local/

# Set paths
KALDI_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
USC_DIR=$KALDI_ROOT/egs/usc

# Source path.sh to set up the environment
source $USC_DIR/path.sh

echo "Creating G.fst following TIMIT procedure..."

# Check if the ARPA models exist
if [ ! -f $USC_DIR/data/local/nist_lm/lm_phone_ug.arpa.gz ] || \
   [ ! -f $USC_DIR/data/local/nist_lm/lm_phone_bg.arpa.gz ]; then
    echo "Error: ARPA models not found in $USC_DIR/data/local/nist_lm/"
    echo "Please run the compile_lm.sh script first."
    exit 1
fi

# Process unigram model
echo "Processing unigram language model..."
utils/format_lm.sh \
    $USC_DIR/data/lang \
    $USC_DIR/data/local/nist_lm/lm_phone_ug.arpa.gz \
    $USC_DIR/data/local/dict/lexicon.txt \
    $USC_DIR/data/lang_ug

# Process bigram model
echo "Processing bigram language model..."
utils/format_lm.sh \
    $USC_DIR/data/lang \
    $USC_DIR/data/local/nist_lm/lm_phone_bg.arpa.gz \
    $USC_DIR/data/local/dict/lexicon.txt \
    $USC_DIR/data/lang_bg

# Create a copy of lang directory with ARPA LM renamed to G.arpa.gz
mkdir -p $USC_DIR/data/lang_test_ug
mkdir -p $USC_DIR/data/lang_test_bg
cp -r $USC_DIR/data/lang/* $USC_DIR/data/lang_test_ug
cp -r $USC_DIR/data/lang/* $USC_DIR/data/lang_test_bg

# Copy the corresponding ARPA LMs
cp $USC_DIR/data/local/nist_lm/lm_phone_ug.arpa.gz $USC_DIR/data/lang_test_ug/G.arpa.gz
cp $USC_DIR/data/local/nist_lm/lm_phone_bg.arpa.gz $USC_DIR/data/lang_test_bg/G.arpa.gz

# Format the language models
echo "Formatting unigram language model..."
utils/format_lm_sri.sh \
    $USC_DIR/data/lang \
    $USC_DIR/data/lang_test_ug/G.arpa.gz \
    $USC_DIR/data/local/dict/lexicon.txt \
    $USC_DIR/data/lang_test_ug

echo "Formatting bigram language model..."
utils/format_lm_sri.sh \
    $USC_DIR/data/lang \
    $USC_DIR/data/lang_test_bg/G.arpa.gz \
    $USC_DIR/data/local/dict/lexicon.txt \
    $USC_DIR/data/lang_test_bg

# Create symbolic links to the formatted language model directories
ln -sf lang_test_ug $USC_DIR/data/lang_test
echo "Created symbolic link: data/lang_test -> data/lang_test_ug"

echo "G.fst creation completed."
echo "Created language model directories:"
echo "  - data/lang_test_ug (unigram model)"
echo "  - data/lang_test_bg (bigram model)"
echo "  - data/lang_test (symbolic link to unigram model)"