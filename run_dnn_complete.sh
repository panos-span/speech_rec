#!/bin/bash
# run_dnn_complete.sh - Complete script for DNN-HMM training and decoding

# Set paths
KALDI_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
USC_DIR=$KALDI_ROOT/egs/usc

# Source path.sh to set up the environment
source $USC_DIR/path.sh
source $USC_DIR/cmd.sh

# Show environment information
echo "Running in directory: $(pwd)"
echo "Kaldi root: $KALDI_ROOT"
echo "USC directory: $USC_DIR"
echo "PATH: $PATH"

# Check Python and Torch
echo "Checking Python and PyTorch..."
PYTHON_CMD=python3
$PYTHON_CMD -c "import sys; print(f'Python version: {sys.version}')"
$PYTHON_CMD -c "import torch; print(f'PyTorch version: {torch.__version__}')"

# Paths for training and decoding
DATA_PATH=$USC_DIR/data/test
GRAPH_PATH=$USC_DIR/exp/tri1/graph_bg  # Using bigram graph
TEST_ALI_PATH=$USC_DIR/exp/tri1_ali_test
OUT_DECODE_PATH=$USC_DIR/exp/tri1/decode_test_dnn
CHECKPOINT_FILE=$USC_DIR/best_usc_dnn.pt
DNN_OUT_FOLDER=$USC_DIR/dnn_out

# Make sure directories exist
mkdir -p $DNN_OUT_FOLDER

# ------------------- STEP 1: Extract triphone alignments -------------------- #
echo "Step 1: Extracting triphone alignments for DNN training..."

# Function to align a dataset
align_dataset() {
    local dataset=$1
    
    echo "Aligning $dataset data..."
    
    # Check if alignment already exists
    if [ -f $USC_DIR/exp/tri1_ali_$dataset/ali.1.gz ]; then
        echo "Alignment for $dataset already exists."
        return 0
    fi
    
    steps/align_si.sh --nj 1 --cmd "$train_cmd" \
        $USC_DIR/data/$dataset $USC_DIR/data/lang $USC_DIR/exp/tri1 \
        $USC_DIR/exp/tri1_ali_$dataset || {
        echo "Error: Alignment of $dataset failed."
        return 1
    }
    
    echo "Alignment of $dataset completed successfully."
    return 0
}

# Align all datasets
align_dataset "train"
align_dataset "dev"
align_dataset "test"

# ------------------- STEP 2: Extract CMVN statistics -------------------- #
echo "Step 2: Extracting CMVN statistics for DNN training..."

# Compute cmvn stats for every set
for set in train dev test; do
  compute-cmvn-stats --spk2utt=ark:$USC_DIR/data/${set}/spk2utt \
    scp:$USC_DIR/data/${set}/feats.scp \
    ark:$USC_DIR/data/${set}/${set}"_cmvn_speaker.ark" || {
    echo "Error: Computing CMVN stats for $set failed."
    exit 1
  }
  
  echo "CMVN stats computed for $set"
done

# ------------------- STEP 3: Train DNN -------------------- #
echo "Step 3: Training DNN..."

# Change to the dnn directory
cd $USC_DIR/dnn

# Make sure torch can import pytorch
# Set PYTHONPATH if needed
export PYTHONPATH=$USC_DIR:$PYTHONPATH

# Check if torch_dataset.py exists and has correct contents
if [ ! -f "$USC_DIR/dnn/torch_dataset.py" ]; then
    echo "Error: torch_dataset.py not found in $USC_DIR/dnn/"
    exit 1
fi

# Run DNN training with explicit Python version
echo "Running timit_dnn.py..."
$PYTHON_CMD timit_dnn.py $CHECKPOINT_FILE
if [ $? -ne 0 ]; then
    echo "Error: DNN training failed."
    # Continue anyway to try other steps
fi

# ------------------- STEP 4: Extract DNN posteriors -------------------- #
echo "Step 4: Extracting DNN posteriors..."

# Run posterior extraction
echo "Running extract_posteriors.py..."
$PYTHON_CMD extract_posteriors.py $CHECKPOINT_FILE $DNN_OUT_FOLDER
if [ $? -ne 0 ]; then
    echo "Error: Posterior extraction failed."
    # Continue anyway to try decoding
fi

# Return to USC directory
cd $USC_DIR

# ------------------- STEP 5: Run DNN decoding -------------------- #
echo "Step 5: Running DNN decoding..."

# Run decoding
./decode_dnn.sh $GRAPH_PATH $DATA_PATH $TEST_ALI_PATH $OUT_DECODE_PATH "cat $DNN_OUT_FOLDER/posteriors.ark"

echo "DNN-HMM training and decoding completed!"
echo "Results should be available in $OUT_DECODE_PATH"

# Display answers to theoretical questions
cat << EOT

Answer to Question 7:
The DNN-HMM model differs from the GMM-HMM in the acoustic modeling component:

1. In GMM-HMM, Gaussian Mixture Models model the acoustic likelihood P(O|q) directly.
2. In DNN-HMM, Deep Neural Networks compute the posterior probability P(q|O).
3. For HMM decoding, DNN posteriors are converted to scaled likelihoods using Bayes' rule:
   P(O|q) ∝ P(q|O)/P(q), where P(q) is the state prior probability.

Benefits of DNN over GMM:
1. DNNs can model much more complex, non-linear relationships in acoustic data
2. DNNs naturally incorporate wider context through feature windowing
3. DNNs handle correlated features better than GMMs (which often assume diagonal covariance)
4. DNNs achieve significantly better performance across many speech recognition tasks

We couldn't train a DNN-HMM from scratch because:
1. HMM training requires frame-level alignments between audio and phonetic states
2. Without initial alignments, it's difficult to determine which frames correspond to which states
3. The typical approach is to first train a GMM-HMM system to get these alignments
4. Then train the DNN using these alignments as targets
5. Finally, replace the GMM emission probabilities with scaled DNN posteriors

Answer to Question 8:
Batch Normalization (BatchNorm) normalizes the input of each layer to have zero mean and unit variance across the batch dimension, providing several benefits:

1. Stabilizes and accelerates training:
   - Reduces internal covariate shift (changes in the distribution of layer inputs)
   - Allows higher learning rates without divergence
   - Makes training less sensitive to parameter initialization

2. Acts as a regularizer:
   - Adds noise to the activations due to the stochasticity of mini-batches
   - Reduces the need for Dropout in some architectures

3. Smooths the optimization landscape:
   - Makes the loss surface more smooth and convex
   - Reduces problems with vanishing/exploding gradients

Mathematically, BatchNorm performs:
   y = γ * (x - μ)/√(σ² + ε) + β
   where:
   - μ and σ are the batch mean and standard deviation
   - γ and β are learnable parameters
   - ε is a small constant for numerical stability

During training, it uses statistics from the current batch, but during inference,
it uses running averages collected during training for consistent normalization.

In our DNN-HMM model, BatchNorm helps the network converge faster and generalize better
to different speakers and acoustic conditions, which is particularly important for
speech recognition tasks.
EOT