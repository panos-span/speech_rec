#!/bin/bash
# setup_usc.sh - Master script for setting up USC speech recognition lab
# Place this in egs/usc/

# Set paths
KALDI_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
USC_DIR=$KALDI_ROOT/egs/usc

echo "Starting USC lab setup..."

# Ensure all the necessary directories exist
mkdir -p $USC_DIR/data/train $USC_DIR/data/dev $USC_DIR/data/test
mkdir -p $USC_DIR/data/local/dict $USC_DIR/data/local/lm_tmp $USC_DIR/data/local/nist_lm
mkdir -p $USC_DIR/data/lang
mkdir -p $USC_DIR/conf

# Create mfcc.conf if it doesn't exist
if [ ! -f $USC_DIR/conf/mfcc.conf ]; then
    echo "--use-energy=false" > $USC_DIR/conf/mfcc.conf
    echo "--sample-frequency=16000" >> $USC_DIR/conf/mfcc.conf
    echo "Created mfcc.conf"
fi

# Create symbolic links if they don't exist
if [ ! -L $USC_DIR/steps ]; then
    ln -sf $KALDI_ROOT/egs/wsj/s5/steps $USC_DIR/steps
    echo "Created symbolic link to steps"
fi

if [ ! -L $USC_DIR/utils ]; then
    ln -sf $KALDI_ROOT/egs/wsj/s5/utils $USC_DIR/utils
    echo "Created symbolic link to utils"
fi

# Create local directory and link to score_kaldi.sh if it doesn't exist
mkdir -p $USC_DIR/local
if [ ! -L $USC_DIR/local/score.sh ]; then
    ln -sf $USC_DIR/steps/score_kaldi.sh $USC_DIR/local/score.sh
    echo "Created symbolic link to score_kaldi.sh"
fi

# Copy cmd.sh and path.sh from wsj if they don't exist
if [ ! -f $USC_DIR/cmd.sh ]; then
    cp $KALDI_ROOT/egs/wsj/s5/cmd.sh $USC_DIR/
    # Modify cmd.sh to use run.pl
    sed -i 's/export train_cmd=.*/export train_cmd=run.pl/' $USC_DIR/cmd.sh
    sed -i 's/export decode_cmd=.*/export decode_cmd=run.pl/' $USC_DIR/cmd.sh
    sed -i 's/export cuda_cmd=.*/export cuda_cmd=run.pl/' $USC_DIR/cmd.sh
    echo "Created cmd.sh"
fi

if [ ! -f $USC_DIR/path.sh ]; then
    cp $KALDI_ROOT/egs/wsj/s5/path.sh $USC_DIR/
    # Modify path.sh to set KALDI_ROOT
    sed -i "s|export KALDI_ROOT=.*|export KALDI_ROOT=$KALDI_ROOT|" $USC_DIR/path.sh
    echo "Created path.sh"
fi

# Run data preparation script
echo "Preparing data..."
bash $USC_DIR/prepare_usc_data.sh

# Run dictionary preparation script
echo "Preparing dictionary..."
bash $USC_DIR/prepare_dict.sh

echo "USC lab setup completed successfully!"