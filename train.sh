#!/bin/bash

# Path variables
KALDI_ROOT=~
EXP=$KALDI_ROOT/egs/usc

# Training monophone models
steps/train_mono.sh --nj 4 --cmd "run.pl" \
  $EXP/data/train $EXP/data/lang $EXP/exp/mono

# Create graph for monophone model
utils/mkgraph.sh $EXP/data/lang_test $EXP/exp/mono $EXP/exp/mono/graph

# Align monophone models
steps/align_si.sh --nj 4 --cmd "run.pl" \
  $EXP/data/train $EXP/data/lang $EXP/exp/mono $EXP/exp/mono_ali

# Train triphone models
steps/train_deltas.sh --cmd "run.pl" \
  2000 10000 $EXP/data/train $EXP/data/lang $EXP/exp/mono_ali $EXP/exp/tri1

# Create graph for triphone model
utils/mkgraph.sh $EXP/data/lang_test $EXP/exp/tri1 $EXP/exp/tri1/graph

# Align triphone models
steps/align_si.sh --nj 4 --cmd "run.pl" \
  $EXP/data/train $EXP/data/lang $EXP/exp/tri1 $EXP/exp/tri1_ali

# Train triphone models with LDA and MLLT
steps/train_lda_mllt.sh --cmd "run.pl" \
  2500 15000 $EXP/data/train $EXP/data/lang $EXP/exp/tri1_ali $EXP/exp/tri2b

# Create graph for LDA+MLLT model
utils/mkgraph.sh $EXP/data/lang_test $EXP/exp/tri2b $EXP/exp/tri2b/graph

# Align triphone models with LDA and MLLT
steps/align_si.sh --nj 4 --cmd "run.pl" \
  $EXP/data/train $EXP/data/lang $EXP/exp/tri2b $EXP/exp/tri2b_ali

# Train triphone models with SAT
steps/train_sat.sh --cmd "run.pl" \
  2500 15000 $EXP/data/train $EXP/data/lang $EXP/exp/tri2b_ali $EXP/exp/tri3b

# Create graph for SAT model
utils/mkgraph.sh $EXP/data/lang_test $EXP/exp/tri3b $EXP/exp/tri3b/graph

echo "Acoustic model training completed"