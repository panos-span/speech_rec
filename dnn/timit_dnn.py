#!/usr/bin/env python3
# Memory efficient version of timit_dnn.py

import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import gc  # Garbage collector

from torch_dataset import TorchSpeechDataset
from torch_dnn import TorchDNN

# Use GPU if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Configuration
NUM_LAYERS = 2
HIDDEN_DIM = 256
USE_BATCH_NORM = True
DROPOUT_P = 0.2
EPOCHS = 50
PATIENCE = 5
BATCH_SIZE = 512

if len(sys.argv) < 2:
    print("USAGE: python timit_dnn.py <PATH/TO/CHECKPOINT_TO_SAVE.pt>")
    sys.exit(1)

BEST_CHECKPOINT = sys.argv[1]

# Alignment directories
TRAIN_ALIGNMENT_DIR = "../exp/tri1_ali_train"
DEV_ALIGNMENT_DIR = "../exp/tri1_ali_dev"
TEST_ALIGNMENT_DIR = "../exp/tri1_ali_test"

def train(model, criterion, optimizer, scheduler, train_loader, dev_loader, epochs=50, patience=3):
    """Train model with early stopping and memory optimizations"""
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += targets.size(0)
            train_correct += (predicted == targets).sum().item()

            # Report progress
            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
            
            # Free memory
            del inputs, targets, outputs, loss
            gc.collect()
        
        train_loss /= train_total
        train_acc = train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in dev_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # Statistics
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()
                
                # Free memory
                del inputs, targets, outputs, loss
                gc.collect()
        
        val_loss /= val_total
        val_acc = val_correct / val_total
        
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Step the scheduler with validation loss
        scheduler.step(val_loss)
        
        # Print current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f"  Current Learning Rate: {current_lr:.6f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            print(f"  Validation loss improved from {best_val_loss:.4f} to {val_loss:.4f}")
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save model state dictionary only
            try:
                print(f"  Saving model to {BEST_CHECKPOINT}...")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_loss': val_loss,
                    'val_acc': val_acc
                }, BEST_CHECKPOINT)
                print("  Model saved successfully")
            except Exception as e:
                print(f"  Error saving model: {e}")
        else:
            patience_counter += 1
            print(f"  No improvement for {patience_counter} epochs")
            
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break
    
    print("Training completed!")
    return model

try:
    # Load datasets
    print("Loading datasets...")
    trainset = TorchSpeechDataset("../", TRAIN_ALIGNMENT_DIR, "train")
    validset = TorchSpeechDataset("../", DEV_ALIGNMENT_DIR, "dev")
    testset = TorchSpeechDataset("../", TEST_ALIGNMENT_DIR, "test")
    
    # Normalize features
    print("Normalizing features...")
    scaler = StandardScaler()
    scaler.fit(trainset.feats)
    trainset.feats = scaler.transform(trainset.feats)
    validset.feats = scaler.transform(validset.feats)
    testset.feats = scaler.transform(testset.feats)
    
    # Get dimensions
    feature_dim = trainset.feats.shape[1]
    n_classes = int(np.max(trainset.labels) + 1)
    
    print(f"Feature dimension: {feature_dim}")
    print(f"Number of classes: {n_classes}")
    
    # Create model
    print("Creating model...")
    dnn = TorchDNN(
        feature_dim,
        n_classes,
        num_layers=NUM_LAYERS,
        batch_norm=USE_BATCH_NORM,
        hidden_dim=HIDDEN_DIM,
        dropout_p=DROPOUT_P
    )
    dnn.to(DEVICE)
    print("Model created successfully")
    
    # Create data loaders
    train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    dev_loader = DataLoader(validset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Define optimizer and loss function
    print("Creating optimizer and loss function...")
    # Define Adam optimizer with better parameters and scheduler
    optimizer = optim.Adam(
        dnn.parameters(),
        lr=0.002,
        weight_decay=1e-5    # L2 regularization to prevent overfitting
    )

    # Learning rate scheduler options
    # Option 1: ReduceLROnPlateau - Reduces learning rate when metric plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',          # Reduce LR when monitored value stops decreasing
        factor=0.1,          # Multiply LR by this factor
        patience=3,         # Number of epochs with no improvement after which LR will be reduced
        verbose=True,        # Print message when LR is reduced
        min_lr=1e-6          # Lower bound on the learning rate
    )

    criterion = nn.CrossEntropyLoss()
    
    # Train model
    print("Starting training...")
    train(dnn, criterion, optimizer, scheduler, train_loader, dev_loader, epochs=EPOCHS, patience=PATIENCE)
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()