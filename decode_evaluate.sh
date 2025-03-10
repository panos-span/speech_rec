#!/bin/bash

# Path variables
KALDI_ROOT=~
EXP=$KALDI_ROOT/egs/usc

# Decode with monophone model
steps/decode.sh --nj 4 --cmd "run.pl" \
  $EXP/exp/mono/graph $EXP/data/test $EXP/exp/mono/decode_test

# Decode with triphone model
steps/decode.sh --nj 4 --cmd "run.pl" \
  $EXP/exp/tri1/graph $EXP/data/test $EXP/exp/tri1/decode_test

# Decode with LDA+MLLT model
steps/decode.sh --nj 4 --cmd "run.pl" \
  $EXP/exp/tri2b/graph $EXP/data/test $EXP/exp/tri2b/decode_test

# Decode with SAT model
steps/decode.sh --nj 4 --cmd "run.pl" \
  $EXP/exp/tri3b/graph $EXP/data/test $EXP/exp/tri3b/decode_test

# Also decode the development set with the best model (SAT)
steps/decode.sh --nj 4 --cmd "run.pl" \
  $EXP/exp/tri3b/graph $EXP/data/dev $EXP/exp/tri3b/decode_dev

# Compute phone error rate (PER)
for dir in mono tri1 tri2b tri3b; do
  grep WER $EXP/exp/$dir/decode_test/wer_* | utils/best_wer.sh
done

echo "Decoding and evaluation completed"