#!/bin/bash
# prepare_lm.sh - Script to prepare language model files
# Place this in egs/usc/

# Set paths
KALDI_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
USC_DIR=$KALDI_ROOT/egs/usc

# Source path.sh to set up the environment
source $USC_DIR/path.sh

echo "Starting language model preparation..."

# Ensure the dictionary directory exists
mkdir -p $USC_DIR/data/local/dict

# Run the Python dictionary preparation script
python3 $USC_DIR/local/prepare_dict.py

echo "Language model preparation completed."
echo "Files created in $USC_DIR/data/local/dict:"
ls -l $USC_DIR/data/local/dict/

# Verify that all required files were created
echo "Verifying required files..."
FILES_OK=true

for file in silence_phones.txt optional_silence.txt nonsilence_phones.txt lexicon.txt extra_questions.txt lm_train.text lm_dev.text lm_test.text; do
    if [ ! -f $USC_DIR/data/local/dict/$file ]; then
        echo "Warning: $file is missing"
        FILES_OK=false
    fi
done

if $FILES_OK; then
    echo "All required files are present."
else
    echo "Some required files are missing."
fi