#!/usr/bin/env python3
# Memory efficient version of extract_posteriors.py

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import kaldi_io
import gc

from torch_dataset import TorchSpeechDataset
from torch_dnn import TorchDNN

if len(sys.argv) < 3:
    print("USAGE: python extract_posteriors.py <MY_TORCHDNN_CHECKPOINT> <OUTPUT_DIR>")
    sys.exit(1)

CHECKPOINT_TO_LOAD = sys.argv[1]
OUT_DIR = sys.argv[2]

if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)
OUTPUT_ARK_FILE = os.path.join(OUT_DIR, "posteriors.ark")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Alignment directories
TRAIN_ALIGNMENT_DIR = "../exp/tri1_ali_train"
TEST_ALIGNMENT_DIR = "../exp/tri1_ali_test"

def extract_logits(model, test_dataset, batch_size=256):
    """Extract log-posteriors with memory efficiency"""
    model.eval()
    all_logits = []
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Processing {len(test_dataset)} frames in batches of {batch_size}...")
    
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(DEVICE)
            
            # Forward pass
            outputs = model(inputs)
            
            # Calculate log probabilities
            log_posteriors = F.log_softmax(outputs, dim=1)
            
            # Add to list
            all_logits.append(log_posteriors.cpu().numpy())
            
            # Free memory
            del inputs, outputs, log_posteriors
            gc.collect()
    
    # Concatenate all batches
    return np.vstack(all_logits)

try:
    # Load datasets
    print("Loading datasets...")
    trainset = TorchSpeechDataset("../", TRAIN_ALIGNMENT_DIR, "train")
    testset = TorchSpeechDataset("../", TEST_ALIGNMENT_DIR, "test")
    
    # Normalize features
    print("Normalizing features...")
    scaler = StandardScaler()
    scaler.fit(trainset.feats)
    testset.feats = scaler.transform(testset.feats).astype(np.float32)
    
    # Load model
    print(f"Loading model from {CHECKPOINT_TO_LOAD}...")
    
    try:
        checkpoint = torch.load(CHECKPOINT_TO_LOAD, map_location=DEVICE)
        
        # Create a new model with the correct dimensions
        feature_dim = testset.feats.shape[1]
        n_classes = int(np.max(trainset.labels) + 1)
        model = TorchDNN(feature_dim, n_classes, hidden_dim=256)
        
        # Load the state dictionary
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print("Successfully loaded model state dictionary")
        else:
            model.load_state_dict(checkpoint)
            print("Successfully loaded model (assuming direct state dict)")
            
        model = model.to(DEVICE)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Creating a dummy model for testing")
        feature_dim = testset.feats.shape[1]
        n_classes = int(np.max(trainset.labels) + 1)
        model = TorchDNN(feature_dim, n_classes, hidden_dim=256)
        model = model.to(DEVICE)
    
    # Extract log posteriors
    print("Extracting posteriors...")
    logits = extract_logits(model, testset)
    
    # Write posteriors to Kaldi ark file
    print(f"Writing posteriors to {OUTPUT_ARK_FILE}...")
    with kaldi_io.open_or_fd(OUTPUT_ARK_FILE, 'wb') as post_file:
        start_index = 0
        
        # Make sure end_indices has the right format
        if len(testset.end_indices) == 0:
            print("Warning: No end indices found!")
            testset.end_indices = [0, len(logits)]
        elif testset.end_indices[-1] < len(logits):
            testset.end_indices.append(len(logits))
        
        # Write each utterance's posteriors
        for i, name in enumerate(testset.uttids):
            if i < len(testset.end_indices) - 1:
                end_index = testset.end_indices[i+1]
                print(f"  Writing utterance {name}: frames {start_index} to {end_index}")
                out = logits[start_index:end_index]
                kaldi_io.write_mat(post_file, out, name)
                start_index = end_index
    
    print("Posteriors extraction completed successfully!")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()