#!/bin/bash
# extract_features_fix.sh - Script to extract MFCCs and compute CMVN stats
# Place this in egs/usc/

# Set paths
KALDI_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
USC_DIR=$KALDI_ROOT/egs/usc

# Source path.sh to set up the environment
source $USC_DIR/path.sh
source $USC_DIR/cmd.sh

echo "Extracting MFCC features and computing CMVN stats..."

# Process each subset
for subset in train dev test; do
    echo "Processing $subset set..."
    
    # Check if the directory exists
    if [ ! -d $USC_DIR/data/$subset ]; then
        echo "Error: Directory $USC_DIR/data/$subset not found"
        continue
    fi

    # Validate the data directory
    echo "Validating data directory for $subset..."
    $USC_DIR/utils/validate_data_dir.sh --no-feats $USC_DIR/data/$subset || {
        echo "Data validation failed for $subset. Attempting to fix..."
        $USC_DIR/utils/fix_data_dir.sh $USC_DIR/data/$subset
    }
    
    # Create logs directory
    mkdir -p $USC_DIR/exp/make_mfcc/$subset/log
    
    # Extract MFCCs - using only 1 job for simplicity
    echo "Extracting MFCCs for $subset set..."
    steps/make_mfcc.sh --nj 1 --cmd "run.pl" \
        $USC_DIR/data/$subset $USC_DIR/exp/make_mfcc/$subset $USC_DIR/data/$subset/mfcc || {
        echo "MFCC extraction failed for $subset. Checking log..."
        cat $USC_DIR/exp/make_mfcc/$subset/log/make_mfcc.*.log
        echo "Continuing to next set..."
        continue
    }
    
    # Compute CMVN stats
    echo "Computing CMVN stats for $subset set..."
    steps/compute_cmvn_stats.sh \
        $USC_DIR/data/$subset $USC_DIR/exp/make_mfcc/$subset $USC_DIR/data/$subset/mfcc || {
        echo "CMVN computation failed for $subset. Checking log..."
        cat $USC_DIR/exp/make_mfcc/$subset/log/compute_cmvn.*.log
        echo "Continuing to next set..."
        continue
    }
    
    echo "Processing of $subset set completed successfully."
done

echo "Feature extraction completed."

# Analyze the first 5 utterances in the training set using Kaldi tools
echo "Analyzing first 5 sentences in the training set..."

# Get the first 5 utterance IDs from the training set
first_5_utts=$(head -n 5 $USC_DIR/data/train/uttids)

# Create a directory for temporary files
mkdir -p $USC_DIR/tmp

# Copy the feats.scp for the first 5 utterances
for utt in $first_5_utts; do
    grep "^$utt " $USC_DIR/data/train/feats.scp >> $USC_DIR/tmp/first_5_utts.scp
done

# Use copy-feats to analyze the features
for utt in $first_5_utts; do
    echo "Analyzing utterance $utt..."
    copy-feats scp:"grep '^$utt ' $USC_DIR/data/train/feats.scp |" ark,t:- 2>&1 | \
        grep -A 2 "copy-feats" | head -3
done

echo "Feature analysis completed."

echo "
Answer to Question 2:
Cepstral Mean and Variance Normalization (CMVN) is a technique used in speech processing to normalize the acoustic features 
to reduce the effects of different recording conditions and speaker variations. Here's a detailed explanation:

Purpose of CMVN:
1. Reduces channel effects: Different microphones and recording environments introduce systematic distortions to the speech signal.
   CMVN helps mitigate these effects by normalizing the feature distributions.

2. Speaker normalization: Different speakers have different vocal tract characteristics, resulting in systematic differences
   in their acoustic features. CMVN helps reduce these speaker-dependent variations.

3. Improves model generalization: By standardizing the input features, CMVN helps the acoustic models generalize better
   to unseen conditions and speakers.

Mathematical formulation:
For each feature dimension i and each frame t in an utterance, CMVN performs:

x̂_t[i] = (x_t[i] - μ[i]) / σ[i]

where:
- x_t[i] is the original feature value
- x̂_t[i] is the normalized feature value
- μ[i] is the mean of feature dimension i across all frames
- σ[i] is the standard deviation of feature dimension i across all frames

The mean and variance are calculated as:
μ[i] = (1/T) * ∑(t=1 to T) x_t[i]
σ²[i] = (1/T) * ∑(t=1 to T) (x_t[i] - μ[i])²

where T is the total number of frames in the utterance.

In Kaldi, CMVN can be applied per-utterance or per-speaker. Per-speaker normalization is more common,
as it provides better estimates of the mean and variance when utterances are short.

The process makes each feature dimension have zero mean and unit variance, which:
1. Makes the features more robust to channel and speaker variations
2. Improves the numerical stability of neural network training
3. Reduces the mismatch between training and test conditions

CMVN is particularly important in real-world speech recognition systems where recording conditions 
vary significantly between training and deployment.

Answer to Question 3:
The number of acoustic frames extracted for each utterance depends on the duration of the audio file 
and the frame shift used in MFCC extraction. In standard Kaldi settings:
- Frame length: 25ms
- Frame shift: 10ms

For the first 5 utterances in the training set, the number of frames should be visible in the output above.

The dimension of the MFCC features in our configuration is 13 (12 cepstral coefficients plus energy). 
This is the default configuration in the mfcc.conf file we created earlier with the parameters:
--use-energy=false
--sample-frequency=16000

If we were to include delta and delta-delta features (which we're not doing in this basic setup), 
the dimension would increase to 39 (13 base features + 13 delta + 13 delta-delta).
"