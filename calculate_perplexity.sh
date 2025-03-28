#!/bin/bash
# calculate_perplexity.sh - Script to calculate perplexity of language models
# Place this in egs/usc/

# Set paths
KALDI_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
USC_DIR=$KALDI_ROOT/egs/usc

# Source path.sh to set up the environment
source $USC_DIR/path.sh

echo "Calculating perplexity of language models..."

# Check if the ARPA models exist
if [ ! -f $USC_DIR/data/local/nist_lm/lm_phone_ug.arpa.gz ] || \
   [ ! -f $USC_DIR/data/local/nist_lm/lm_phone_bg.arpa.gz ]; then
    echo "Error: ARPA models not found in $USC_DIR/data/local/nist_lm/"
    echo "Please run the compile_lm.sh script first."
    exit 1
fi

# Prepare test data for perplexity calculation
echo "Preparing test data..."
for subset in dev test; do
    # Extract just the phoneme sequences (without utterance IDs)
    cat $USC_DIR/data/$subset/text | cut -d' ' -f2- > $USC_DIR/data/local/lm_tmp/${subset}_text
done

# Calculate perplexity for unigram model using ngram command from SRILM
echo "Calculating perplexity for unigram model..."
echo "On validation set (dev):"
ngram -lm $USC_DIR/data/local/nist_lm/lm_phone_ug.arpa.gz -ppl $USC_DIR/data/local/lm_tmp/dev_text

echo "On test set:"
ngram -lm $USC_DIR/data/local/nist_lm/lm_phone_ug.arpa.gz -ppl $USC_DIR/data/local/lm_tmp/test_text

# Calculate perplexity for bigram model
echo "Calculating perplexity for bigram model..."
echo "On validation set (dev):"
ngram -lm $USC_DIR/data/local/nist_lm/lm_phone_bg.arpa.gz -ppl $USC_DIR/data/local/lm_tmp/dev_text

echo "On test set:"
ngram -lm $USC_DIR/data/local/nist_lm/lm_phone_bg.arpa.gz -ppl $USC_DIR/data/local/lm_tmp/test_text

echo "Perplexity calculation completed."
echo "
Answer to Question 1:
The perplexity values indicate how well our language models predict the phoneme sequences in the validation and test sets. 
Lower perplexity values indicate better predictive power of the model.

The perplexity of a language model on test data is defined as the exponentiated average negative log-likelihood per word.
It can be interpreted as the weighted average branching factor of the language model, or how many choices on average 
the model is uncertain about at each step.

We expect the bigram model to have lower perplexity than the unigram model, as it captures more context by 
modeling phoneme pairs rather than just individual phonemes. Lower perplexity typically correlates with better 
recognition performance, as the model is more effective at predicting the actual phoneme sequences.
"