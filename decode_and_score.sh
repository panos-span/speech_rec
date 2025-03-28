#!/bin/bash
# decode_and_score.sh - Script to decode and score validation and test data
# Place this in egs/usc/

# Set paths
KALDI_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
USC_DIR=$KALDI_ROOT/egs/usc

# Source path.sh to set up the environment
source $USC_DIR/path.sh
source $USC_DIR/cmd.sh

echo "Decoding validation and test data with Viterbi algorithm..."

# Check if HCLG graphs exist
if [ ! -f $USC_DIR/exp/mono/graph_ug/HCLG.fst ] || [ ! -f $USC_DIR/exp/mono/graph_bg/HCLG.fst ]; then
    echo "Error: HCLG graphs not found. Please create them first."
    exit 1
fi

# Function to decode a dataset with a specific language model
decode_dataset() {
    local dataset=$1   # dev or test
    local lm_type=$2   # ug or bg

    echo "Decoding $dataset data with ${lm_type}ram language model..."
    
    steps/decode.sh --nj 1 --cmd "$decode_cmd" \
        $USC_DIR/exp/mono/graph_${lm_type} \
        $USC_DIR/data/${dataset} \
        $USC_DIR/exp/mono/decode_${lm_type}_${dataset} || {
        echo "Error: Decoding ${dataset} with ${lm_type}ram failed."
        return 1
    }
    
    # Score the decoded results
    echo "Scoring ${dataset} results with ${lm_type}ram language model..."
    steps/score_kaldi.sh --cmd "$decode_cmd" \
        $USC_DIR/data/${dataset} \
        $USC_DIR/exp/mono/graph_${lm_type} \
        $USC_DIR/exp/mono/decode_${lm_type}_${dataset} || {
        echo "Error: Scoring ${dataset} with ${lm_type}ram failed."
        return 1
    }
    
    echo "Decoding and scoring of ${dataset} with ${lm_type}ram completed."
    return 0
}

# Decode dev and test with unigram
decode_dataset "dev" "ug"
decode_dataset "test" "ug"

# Decode dev and test with bigram
decode_dataset "dev" "bg"
decode_dataset "test" "bg"

echo "All decoding and scoring completed."

# Function to display PER results
display_results() {
    local dataset=$1   # dev or test
    local lm_type=$2   # ug or bg
    
    echo "=== PER Results for ${dataset} with ${lm_type}ram language model ==="
    
    # Get the best WER result
    if [ -f $USC_DIR/exp/mono/decode_${lm_type}_${dataset}/scoring_kaldi/best_wer ]; then
        best_wer=$(cat $USC_DIR/exp/mono/decode_${lm_type}_${dataset}/scoring_kaldi/best_wer)
        echo "Best WER: $best_wer"
        
        # Extract the scoring parameters
        if [[ $best_wer =~ wer_([0-9]+)_([0-9\.]+) ]]; then
            beam_width=${BASH_REMATCH[1]}
            word_ins_penalty=${BASH_REMATCH[2]}
            echo "Hyperparameters: beam_width=$beam_width, word_insertion_penalty=$word_ins_penalty"
        fi
        
        # Get the detailed scoring breakdown
        if [ -f $USC_DIR/exp/mono/decode_${lm_type}_${dataset}/scoring_kaldi/wer_${beam_width}_${word_ins_penalty} ]; then
            cat $USC_DIR/exp/mono/decode_${lm_type}_${dataset}/scoring_kaldi/wer_${beam_width}_${word_ins_penalty}
        fi
    else
        echo "No scoring results found."
    fi
    
    echo ""
}

# Display results for all configurations
display_results "dev" "ug"
display_results "test" "ug"
display_results "dev" "bg"
display_results "test" "bg"

echo "
The two hyperparameters in the scoring process are:

1. beam_width (or 'beam'): Controls the beam search width during decoding. 
   A larger beam width allows the decoder to explore more hypotheses, potentially 
   finding better paths but at the cost of increased computation.

2. word_insertion_penalty (or 'wip'): A parameter that controls the tradeoff 
   between insertions and deletions. A higher value penalizes word insertions more, 
   which generally reduces insertions but may increase deletions.

The Phone Error Rate (PER) is calculated as:
PER = 100 * (insertions + substitutions + deletions) / #phonemes

where #phonemes is the total number of phonemes in the reference transcriptions.

The best values for these hyperparameters in our experiments are shown above
in the results for each configuration.
"