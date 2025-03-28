#!/bin/bash
# compile_lm.sh - Script to compile language models to ARPA format
# Place this in egs/usc/

# Set paths
KALDI_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
USC_DIR=$KALDI_ROOT/egs/usc

# Source path.sh to set up the environment
source $USC_DIR/path.sh

echo "Compiling language models to ARPA format..."

# Ensure the nist_lm directory exists
mkdir -p $USC_DIR/data/local/nist_lm

# Check if the intermediate language models exist
if [ ! -f $USC_DIR/data/local/lm_tmp/lm_phone_ug.ilm.gz ]; then
    echo "Error: Unigram model lm_phone_ug.ilm.gz not found in $USC_DIR/data/local/lm_tmp/"
    echo "Please run the build_lm.sh script first."
    exit 1
fi

if [ ! -f $USC_DIR/data/local/lm_tmp/lm_phone_bg.ilm.gz ]; then
    echo "Error: Bigram model lm_phone_bg.ilm.gz not found in $USC_DIR/data/local/lm_tmp/"
    echo "Please run the build_lm.sh script first."
    exit 1
fi

# Compile unigram model to ARPA format
echo "Compiling unigram model to ARPA format..."
compile-lm $USC_DIR/data/local/lm_tmp/lm_phone_ug.ilm.gz -t=yes /dev/stdout | grep -v unk | gzip -c > $USC_DIR/data/local/nist_lm/lm_phone_ug.arpa.gz

# Check if unigram ARPA model was created successfully
if [ ! -f $USC_DIR/data/local/nist_lm/lm_phone_ug.arpa.gz ]; then
    echo "Error: Failed to compile unigram model to ARPA format."
    exit 1
fi

echo "Unigram ARPA model created successfully."

# Compile bigram model to ARPA format
echo "Compiling bigram model to ARPA format..."
compile-lm $USC_DIR/data/local/lm_tmp/lm_phone_bg.ilm.gz -t=yes /dev/stdout | grep -v unk | gzip -c > $USC_DIR/data/local/nist_lm/lm_phone_bg.arpa.gz

# Check if bigram ARPA model was created successfully
if [ ! -f $USC_DIR/data/local/nist_lm/lm_phone_bg.arpa.gz ]; then
    echo "Error: Failed to compile bigram model to ARPA format."
    exit 1
fi

echo "Bigram ARPA model created successfully."

echo "Language models compiled to ARPA format successfully."
echo "Files created in $USC_DIR/data/local/nist_lm:"
ls -lh $USC_DIR/data/local/nist_lm/

echo "Next step: Create the FST of the lexicon (L.fst) using prepare_lang.sh."