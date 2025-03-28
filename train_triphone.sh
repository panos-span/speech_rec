#!/bin/bash
# train_triphone.sh - Script to train a triphone model and evaluate it
# Place this in egs/usc/

# Set paths
KALDI_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
USC_DIR=$KALDI_ROOT/egs/usc

# Source path.sh to set up the environment
source $USC_DIR/path.sh
source $USC_DIR/cmd.sh

echo "Starting triphone model training process..."

# Step 1: Align phonemes using the monophone model
echo "Aligning phonemes using monophone model..."
steps/align_si.sh --nj 1 --cmd "$train_cmd" \
    $USC_DIR/data/train $USC_DIR/data/lang $USC_DIR/exp/mono $USC_DIR/exp/mono_ali || {
    echo "Error: Phoneme alignment failed."
    exit 1
}

# Step 2: Train triphone model using alignments
echo "Training triphone model..."
steps/train_deltas.sh --cmd "$train_cmd" \
    2500 15000 $USC_DIR/data/train $USC_DIR/data/lang $USC_DIR/exp/mono_ali $USC_DIR/exp/tri1 || {
    echo "Error: Triphone training failed."
    exit 1
}

# Step 3: Create HCLG graphs for triphone model with both unigram and bigram
echo "Creating HCLG graphs for triphone model..."

# Unigram graph
if [ -d $USC_DIR/data/lang_ug ]; then
    LANG_UG_DIR=$USC_DIR/data/lang_ug
elif [ -d $USC_DIR/data/lang_ug_tmp ]; then
    LANG_UG_DIR=$USC_DIR/data/lang_ug_tmp
else
    echo "Error: Unigram language model directory not found."
    exit 1
fi

utils/mkgraph.sh $LANG_UG_DIR $USC_DIR/exp/tri1 $USC_DIR/exp/tri1/graph_ug || {
    echo "Error: Failed to create unigram HCLG graph for triphone model."
    exit 1
}

# Bigram graph
if [ -d $USC_DIR/data/lang_bg ]; then
    LANG_BG_DIR=$USC_DIR/data/lang_bg
elif [ -d $USC_DIR/data/lang_bg_tmp ]; then
    LANG_BG_DIR=$USC_DIR/data/lang_bg_tmp
else
    echo "Error: Bigram language model directory not found."
    exit 1
fi

utils/mkgraph.sh $LANG_BG_DIR $USC_DIR/exp/tri1 $USC_DIR/exp/tri1/graph_bg || {
    echo "Error: Failed to create bigram HCLG graph for triphone model."
    exit 1
}

# Step 4: Decode and score validation and test sets
echo "Decoding and scoring with triphone model..."

# Function to decode and score a dataset with a specific language model
decode_and_score() {
    local dataset=$1   # dev or test
    local lm_type=$2   # ug or bg

    echo "Decoding $dataset data with ${lm_type}ram language model (triphone)..."
    
    # Decode
    steps/decode.sh --nj 1 --cmd "$decode_cmd" \
        $USC_DIR/exp/tri1/graph_${lm_type} \
        $USC_DIR/data/${dataset} \
        $USC_DIR/exp/tri1/decode_${lm_type}_${dataset} || {
        echo "Error: Triphone decoding of ${dataset} with ${lm_type}ram failed."
        return 1
    }
    
    # Score
    steps/score_kaldi.sh --cmd "$decode_cmd" \
        $USC_DIR/data/${dataset} \
        $USC_DIR/exp/tri1/graph_${lm_type} \
        $USC_DIR/exp/tri1/decode_${lm_type}_${dataset} || {
        echo "Error: Triphone scoring of ${dataset} with ${lm_type}ram failed."
        return 1
    }
    
    echo "Triphone decoding and scoring of ${dataset} with ${lm_type}ram completed."
    return 0
}

# Decode and score all combinations
decode_and_score "dev" "ug"
decode_and_score "test" "ug"
decode_and_score "dev" "bg"
decode_and_score "test" "bg"

echo "All triphone decoding and scoring completed."

# Display PER results
for dataset in dev test; do
    for lm_type in ug bg; do
        echo "=== Triphone PER Results for ${dataset} with ${lm_type}ram language model ==="
        
        # Get the best WER result
        if [ -f $USC_DIR/exp/tri1/decode_${lm_type}_${dataset}/scoring_kaldi/best_wer ]; then
            best_wer=$(cat $USC_DIR/exp/tri1/decode_${lm_type}_${dataset}/scoring_kaldi/best_wer)
            echo "Best WER: $best_wer"
            
            # Extract the scoring parameters
            if [[ $best_wer =~ wer_([0-9]+)_([0-9\.]+) ]]; then
                beam_width=${BASH_REMATCH[1]}
                word_ins_penalty=${BASH_REMATCH[2]}
                echo "Hyperparameters: beam_width=$beam_width, word_insertion_penalty=$word_ins_penalty"
                
                # Get the detailed scoring breakdown
                if [ -f $USC_DIR/exp/tri1/decode_${lm_type}_${dataset}/scoring_kaldi/wer_${beam_width}_${word_ins_penalty} ]; then
                    cat $USC_DIR/exp/tri1/decode_${lm_type}_${dataset}/scoring_kaldi/wer_${beam_width}_${word_ins_penalty}
                fi
            fi
        else
            echo "No scoring results found."
        fi
        
        echo ""
    done
done

echo "Triphone model training and evaluation completed."
echo "
Answer to Question 4:
A GMM-HMM (Gaussian Mixture Model - Hidden Markov Model) acoustic model combines two 
statistical modeling techniques to represent speech sounds.

Structure of a GMM-HMM acoustic model:
- Each phoneme is typically modeled by a 3-state left-to-right HMM
- Each state in the HMM is associated with a GMM emission probability
- For triphones, the model accounts for context by modeling each phoneme in the context of its neighbors

Role of HMMs (Markov models):
1. HMMs model the temporal dynamics of speech sounds
2. They handle the variable duration of phonemes through self-transitions
3. They represent the sequential nature of speech through transition probabilities
4. They allow for efficient decoding using dynamic programming (Viterbi algorithm)

Role of GMMs (Gaussian mixtures):
1. GMMs model the distribution of acoustic features for each HMM state
2. They can represent complex, multi-modal distributions by using multiple Gaussian components
3. Each Gaussian component has a weight, mean vector, and covariance matrix
4. GMMs provide the emission probabilities: p(acoustic_feature | state)

Training process of a GMM-HMM model:
1. Initialization: Start with flat-start monophone models where all states share the same GMM
2. Re-estimation: Use Baum-Welch algorithm (a form of Expectation-Maximization) to:
   - E-step: Compute state occupancy probabilities given current model parameters
   - M-step: Update model parameters to maximize likelihood of the data
3. Iterative refinement: Gradually increase model complexity by:
   - Increasing the number of Gaussian components per state
   - Splitting existing Gaussians and re-estimating parameters
4. For monophone training specifically:
   - Initialize with a single Gaussian per state for all phones
   - Align data to states using Viterbi alignment
   - Re-estimate parameters
   - Increase number of Gaussians and repeat
   - The process typically converges after 30-40 iterations

The training process for a monophone model is simpler than for triphones since it doesn't 
need to handle context dependency, but follows the same general approach of alternating 
between alignment and parameter estimation.

Answer to Question 5:
In speech recognition, we want to find the most likely sequence of words (or phonemes) 
given the acoustic observations. This is expressed using Bayes' theorem as:

W* = argmax_W P(W|O) = argmax_W [P(O|W) × P(W)] / P(O)

Where:
- W* is the most likely word/phoneme sequence
- O is the sequence of acoustic observations (MFCC features)
- P(W|O) is the posterior probability we want to maximize
- P(O|W) is the acoustic model probability (likelihood of observations given words)
- P(W) is the prior probability from the language model
- P(O) is the probability of the observations (constant for all W, so can be ignored)

Simplified, we have:
W* = argmax_W P(O|W) × P(W)

For phoneme recognition:
1. The acoustic model P(O|W) is computed using the GMM-HMM model:
   - HMM provides the state sequence probability
   - GMM provides the observation probability given each state
2. The language model P(W) gives the prior probability of the phoneme sequence
3. The Viterbi algorithm efficiently finds the most likely sequence by:
   - Computing the probability of the best path to each state at each time step
   - Keeping track of the path that led to that best probability
   - Tracing back to find the most likely sequence

In Kaldi, this process is implemented through the HCLG decoding graph, which 
combines all the necessary information for efficient decoding.

Answer to Question 6:
The HCLG graph in Kaldi is a weighted finite state transducer (WFST) that combines 
four different transducers into one optimized search network:

H: The HMM transducer
- Represents the topology of the HMM for each context-dependent phone
- Maps from transition-ids to context-dependent phones (pdf-ids)
- Encodes the self-loops and transitions between states

C: The context-dependency transducer
- Maps from context-dependent phones to context-independent phones
- For triphones, it maps a phone in context (e.g., a-b+c) to the base phone (b)
- Handles the sharing of parameters through phonetic decision trees

L: The lexicon transducer
- Maps from words to phones
- Incorporates pronunciation variants
- Includes special symbols for handling silence and word boundaries

G: The grammar transducer
- Represents the language model (n-gram probabilities)
- Maps between word sequences
- Encodes the probability of word sequences

The composition of these transducers (HCLG = H ◦ C ◦ L ◦ G) creates a single 
optimized graph where:
- Input labels are the transition-ids from the acoustic model
- Output labels are the words
- Weights combine acoustic and language model probabilities

This unified graph allows for efficient one-pass decoding where all constraints 
(acoustic, phonetic, lexical, and grammatical) are applied simultaneously during 
the search. The graph is optimized through operations like determinization, 
minimization, and weight pushing to make decoding faster and more efficient.

During decoding, Kaldi uses a token-passing algorithm on this graph to find the 
most likely word sequence given the acoustic observations.
"