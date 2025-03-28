#!/bin/bash
# prepare_lang_fst.sh - Script to create FST of the lexicon (L.fst)
# Place this in egs/usc/

# Set paths
KALDI_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
USC_DIR=$KALDI_ROOT/egs/usc

# Source path.sh to set up the environment
source $USC_DIR/path.sh

echo "Creating FST of the lexicon (L.fst) using prepare_lang.sh..."

# Explicitly add Kaldi's bin and fstbin directories to the PATH
export PATH=$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/src/fstbin:$KALDI_ROOT/src/bin:$PATH

# Check if the FST tools are available
if ! command -v fstaddselfloops &> /dev/null; then
    echo "Error: fstaddselfloops command not found. Trying to locate it..."
    find $KALDI_ROOT -name "fstaddselfloops" -type f
    
    # Check if openfst is properly installed
    if [ ! -d "$KALDI_ROOT/tools/openfst" ]; then
        echo "Error: OpenFst not found in $KALDI_ROOT/tools/openfst"
        echo "Please make sure Kaldi is properly installed with OpenFst."
        exit 1
    fi
    
    echo "Error: Cannot find fstaddselfloops command."
    echo "Make sure Kaldi and OpenFst are properly installed."
    echo "You might need to recompile Kaldi with: cd $KALDI_ROOT/src && make -j clean depend; make -j"
    exit 1
fi

# Check if dictionary files exist
if [ ! -f $USC_DIR/data/local/dict/lexicon.txt ] || \
   [ ! -f $USC_DIR/data/local/dict/nonsilence_phones.txt ] || \
   [ ! -f $USC_DIR/data/local/dict/silence_phones.txt ] || \
   [ ! -f $USC_DIR/data/local/dict/optional_silence.txt ]; then
    echo "Error: Dictionary files not found in $USC_DIR/data/local/dict/"
    echo "Please run the dictionary preparation script first."
    exit 1
fi

# Create a temporary directory for prepare_lang.sh
mkdir -p $USC_DIR/data/local/lang_tmp

# Run prepare_lang.sh
echo "Running prepare_lang.sh..."
$USC_DIR/utils/prepare_lang.sh \
    $USC_DIR/data/local/dict \
    "sil" \
    $USC_DIR/data/local/lang_tmp \
    $USC_DIR/data/lang

# Check if L.fst was created successfully
if [ ! -f $USC_DIR/data/lang/L.fst ]; then
    echo "Error: Failed to create L.fst."
    exit 1
fi

# Check if the FST file is valid
if command -v fstinfo &> /dev/null; then
    echo "Validating L.fst..."
    if ! fstinfo $USC_DIR/data/lang/L.fst &> /dev/null; then
        echo "Warning: L.fst might be invalid. FST validation failed."
    else
        echo "L.fst validation successful."
    fi
fi

echo "FST of the lexicon (L.fst) created successfully."
echo "Files created in $USC_DIR/data/lang:"
ls -l $USC_DIR/data/lang/ | head -10
echo "... (more files)"

echo "Next step: Sort the data files and create spk2utt."