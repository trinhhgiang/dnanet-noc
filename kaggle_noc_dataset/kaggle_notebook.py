"""
Kaggle Notebook Script: DNA Mixture Number of Contributors (NoC) Classification
==================================================================================

This script demonstrates end-to-end workflow for the NoC classifier:
1. Load converted PROVEDIt data from the Kaggle dataset
2. Prepare features and labels
3. Train the NoC classifier
4. Evaluate on test set
5. Generate predictions and visualizations

Usage in Kaggle Notebook:
    %run kaggle_notebook.py
    or
    exec(open('kaggle_notebook.py').read())

Requirements:
    - torch
    - scikit-learn
    - numpy
    - pandas
    - matplotlib
    - seaborn
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


# ============================================================================
# 1. CONFIGURATION
# ============================================================================

class Config:
    """Configuration for Kaggle notebook execution"""
    
    # Paths (adjust for your Kaggle environment)
    DATASET_PATH = "/kaggle/input/dnanet-noc-mixture/"  # Dataset root
    OUTPUT_PATH = "/kaggle/output/"  # Kaggle output directory
    
    # If running locally for testing:
    # DATASET_PATH = "../input/dnanet-noc-mixture/"
    # OUTPUT_PATH = "./"
    
    # Model hyperparameters
    HIDDEN_DIM = 128
    NUM_CLASSES = 10  # 1-10 contributors
    DROPOUT_RATE = 0.2
    LEARNING_RATE = 0.001
    BATCH_SIZE = 8
    EPOCHS = 50
    EARLY_STOPPING_PATIENCE = 10
    SEED = 42
    TEST_SIZE = 0.2
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================================
# 2. DATASET LOADING
# ============================================================================

def load_converted_dataset(dataset_path: str) -> tuple:
    """
    Load pre-converted PROVEDIt dataset from Kaggle.
    
    Returns:
        (features, labels, feature_names)
        - features: np.ndarray (n_samples, n_features)
        - labels: np.ndarray (n_samples,) with NoC (1-10)
        - feature_names: list of feature names
    """
    print(f"📂 Loading dataset from: {dataset_path}")
    
    # Load features
    features_path = os.path.join(dataset_path, "features.npy")
    labels_path = os.path.join(dataset_path, "labels.npy")
    feature_names_path = os.path.join(dataset_path, "feature_names.json")
    
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"features.npy not found at {features_path}")
    
    features = np.load(features_path)
    labels = np.load(labels_path)
    
    with open(feature_names_path, 'r') as f:
        feature_names = json.load(f)
    
    print(f"✓ Loaded {features.shape[0]} samples with {features.shape[1]} features")
    print(f"✓ NoC distribution: {np.bincount(labels)}")
    
    return features, labels, feature_names


# ============================================================================
# 3. NEURAL NETWORK MODEL
# ============================================================================

class NoCClassifier(nn.Module):
    """Fully-connected neural network for NoC classification"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, 
                 num_classes: int = 10, dropout_rate: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(hidden_dim, 64)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(64, num_classes)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        return x
    
    def predict(self, x):
        """Return class predictions (training off)"""
        self.eval()
        with torch.no_grad():
            logits = self(x)
            return torch.argmax(logits, dim=1) + 1  # NoC is 1-10
    
    def predict_proba(self, x):
        """Return probability distributions"""
        self.eval()
        with torch.no_grad():
            logits = self(x)
            probs = torch.softmax(logits, dim=1)
            return probs


# ============================================================================
# 4. TRAINING LOOP
# ============================================================================

def train_model(model: nn.Module, train_loader, val_loader, 
                epochs: int, learning_rate: float, device: str, 
                patience: int = 10) -> dict:
    """
    Train the model with early stopping.
    
    Returns:
        history: dict with 'train_loss', 'val_loss', 'val_acc', 'best_epoch'
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model = model.to(device)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_acc': [],
        'best_epoch': 0,
        'best_val_acc': 0.0
    }
    
    patience_counter = 0
    
    print(f"\n🚀 Training on {device}...")
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_x.shape[0]
        
        train_loss /= len(train_loader.dataset)
        history['train_loss'].append(train_loss)
        
        # Validate
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                val_loss += loss.item() * batch_x.shape[0]
                
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == batch_y).sum().item()
                val_total += batch_y.shape[0]
        
        val_loss /= len(val_loader.dataset)
        val_acc = val_correct / val_total
        
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print progress
        if (epoch + 1) % max(1, epochs // 10) == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val Acc: {val_acc:.4f}")
        
        # Early stopping
        if val_acc > history['best_val_acc']:
            history['best_val_acc'] = val_acc
            history['best_epoch'] = epoch + 1
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  ⏹️  Early stopping at epoch {epoch+1}")
                break
    
    print(f"✓ Training complete. Best validation accuracy: {history['best_val_acc']:.4f}")
    return history


# ============================================================================
# 5. EVALUATION & METRICS
# ============================================================================

def evaluate_model(model: nn.Module, test_loader, device: str) -> dict:
    """Evaluate model on test set"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            preds = torch.argmax(logits, dim=1) + 1  # NoC: 1-10
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend((batch_y.numpy() + 1))  # Convert to 1-10
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    accuracy = accuracy_score(all_labels, all_preds)
    
    results = {
        'accuracy': accuracy,
        'predictions': all_preds,
        'labels': all_labels,
        'confusion_matrix': confusion_matrix(all_labels, all_preds, 
                                             labels=range(1, 11)),
        'classification_report': classification_report(all_labels, all_preds, 
                                                       labels=range(1, 11),
                                                       zero_division=0)
    }
    
    return results


# ============================================================================
# 6. VISUALIZATION
# ============================================================================

def plot_training_history(history: dict, output_path: str):
    """Plot training curves"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss
    axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0].plot(history['val_loss'], label='Val Loss', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training & Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(history['val_acc'], label='Val Accuracy', marker='o', color='green')
    axes[1].axhline(y=history['best_val_acc'], color='r', linestyle='--', 
                    label=f"Best: {history['best_val_acc']:.4f}")
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'training_curves.png'), dpi=100, bbox_inches='tight')
    print(f"✓ Saved training_curves.png")


def plot_confusion_matrix(cm: np.ndarray, output_path: str):
    """Plot confusion matrix"""
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=range(1, 11), yticklabels=range(1, 11))
    ax.set_xlabel('Predicted NoC')
    ax.set_ylabel('True NoC')
    ax.set_title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'confusion_matrix.png'), dpi=100, bbox_inches='tight')
    print(f"✓ Saved confusion_matrix.png")


def plot_noc_distribution(labels: np.ndarray, output_path: str):
    """Plot NoC label distribution"""
    fig, ax = plt.subplots(figsize=(10, 5))
    noc_counts = np.bincount(labels, minlength=11)[1:]
    bars = ax.bar(range(1, 11), noc_counts, color='steelblue', edgecolor='black')
    ax.set_xlabel('Number of Contributors (NoC)')
    ax.set_ylabel('Count')
    ax.set_title('Dataset NoC Distribution')
    ax.set_xticks(range(1, 11))
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'noc_distribution.png'), dpi=100, bbox_inches='tight')
    print(f"✓ Saved noc_distribution.png")


# ============================================================================
# 7. MAIN EXECUTION
# ============================================================================

def main():
    """Main execution flow"""
    
    print("=" * 80)
    print("🧬 DNA Mixture NoC Classification - Kaggle Notebook")
    print("=" * 80)
    
    # Set seed for reproducibility
    np.random.seed(Config.SEED)
    torch.manual_seed(Config.SEED)
    
    # Create output directory
    os.makedirs(Config.OUTPUT_PATH, exist_ok=True)
    
    # ---- LOAD DATA ----
    features, labels, feature_names = load_converted_dataset(Config.DATASET_PATH)
    
    # Convert labels from 1-10 to 0-9 for PyTorch
    labels_pytorch = labels - 1
    
    # Plot NoC distribution
    plot_noc_distribution(labels_pytorch, Config.OUTPUT_PATH)
    
    # ---- PREPARE DATA ----
    print(f"\n📊 Preparing data (test_size={Config.TEST_SIZE})...")
    
    # Normalize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Train-test split (stratified by NoC)
    X_train, X_test, y_train, y_test = train_test_split(
        features_scaled, labels_pytorch,
        test_size=Config.TEST_SIZE,
        random_state=Config.SEED,
        stratify=labels_pytorch
    )
    
    # Further split train into train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=0.2,
        random_state=Config.SEED,
        stratify=y_train
    )
    
    print(f"  Train samples: {X_train.shape[0]}")
    print(f"  Val samples: {X_val.shape[0]}")
    print(f"  Test samples: {X_test.shape[0]}")
    
    # Create data loaders
    from torch.utils.data import TensorDataset, DataLoader
    
    train_dataset = TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(y_train).long()
    )
    val_dataset = TensorDataset(
        torch.from_numpy(X_val).float(),
        torch.from_numpy(y_val).long()
    )
    test_dataset = TensorDataset(
        torch.from_numpy(X_test).float(),
        torch.from_numpy(y_test).long()
    )
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE)
    
    # ---- BUILD MODEL ----
    print(f"\n🏗️  Building model...")
    model = NoCClassifier(
        input_dim=features.shape[1],
        hidden_dim=Config.HIDDEN_DIM,
        num_classes=Config.NUM_CLASSES,
        dropout_rate=Config.DROPOUT_RATE
    )
    print(f"✓ Model architecture:\n{model}")
    
    # ---- TRAIN MODEL ----
    history = train_model(
        model, train_loader, val_loader,
        epochs=Config.EPOCHS,
        learning_rate=Config.LEARNING_RATE,
        device=Config.DEVICE,
        patience=Config.EARLY_STOPPING_PATIENCE
    )
    
    # ---- EVALUATE MODEL ----
    print(f"\n📈 Evaluating on test set...")
    results = evaluate_model(model, test_loader, Config.DEVICE)
    
    print(f"\n✓ Test Accuracy: {results['accuracy']:.4f}")
    print(f"\nClassification Report:")
    print(results['classification_report'])
    
    # ---- SAVE RESULTS ----
    print(f"\n💾 Saving results...")
    
    # Save model
    torch.save(model.state_dict(), os.path.join(Config.OUTPUT_PATH, 'model.pt'))
    
    # Save metrics as JSON
    metrics = {
        'test_accuracy': float(results['accuracy']),
        'best_val_accuracy': float(history['best_val_acc']),
        'best_epoch': history['best_epoch'],
        'config': {
            'hidden_dim': Config.HIDDEN_DIM,
            'num_classes': Config.NUM_CLASSES,
            'learning_rate': Config.LEARNING_RATE,
            'batch_size': Config.BATCH_SIZE,
            'epochs': Config.EPOCHS,
            'test_size': Config.TEST_SIZE
        },
        'timestamp': datetime.now().isoformat()
    }
    
    with open(os.path.join(Config.OUTPUT_PATH, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'label': results['labels'],
        'prediction': results['predictions']
    })
    predictions_df.to_csv(os.path.join(Config.OUTPUT_PATH, 'predictions.csv'), index=False)
    
    # Save feature names for reference
    with open(os.path.join(Config.OUTPUT_PATH, 'feature_names.json'), 'w') as f:
        json.dump(feature_names, f, indent=2)
    
    print(f"✓ All results saved to {Config.OUTPUT_PATH}")
    
    # ---- VISUALIZATIONS ----
    print(f"\n🎨 Generating visualizations...")
    plot_training_history(history, Config.OUTPUT_PATH)
    plot_confusion_matrix(results['confusion_matrix'], Config.OUTPUT_PATH)
    
    print(f"\n" + "=" * 80)
    print("✅ Kaggle notebook execution complete!")
    print("=" * 80)
    
    return model, history, results


if __name__ == '__main__':
    model, history, results = main()
