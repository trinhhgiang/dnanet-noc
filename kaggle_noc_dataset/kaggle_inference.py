"""
Kaggle Inference Script: Use trained NoC classifier on new data
================================================================

This script loads a trained NoC model and makes predictions on new DNA mixture samples.

Usage in Kaggle Notebook:
    from kaggle_inference import NoCPredictor
    predictor = NoCPredictor('path/to/model.pt', 'path/to/feature_names.json')
    predictions = predictor.predict(features_array)

"""

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from typing import Union, Tuple, List
from sklearn.preprocessing import StandardScaler


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
    
    def predict_proba(self, x):
        """Return probability distributions"""
        self.eval()
        with torch.no_grad():
            logits = self(x)
            probs = torch.softmax(logits, dim=1)
            return probs


class NoCPredictor:
    """Wrapper for easy NoC prediction on new samples"""
    
    def __init__(self, model_path: str, feature_names_path: str = None, 
                 scaler_path: str = None, device: str = None):
        """
        Initialize predictor with trained model.
        
        Args:
            model_path: Path to saved model.pt
            feature_names_path: Path to feature_names.json (optional)
            scaler_path: Path to saved scaler (optional)
            device: 'cuda' or 'cpu' (auto-detected if None)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_names = []
        self.scaler = None
        
        # Load feature names
        if feature_names_path and Path(feature_names_path).exists():
            with open(feature_names_path, 'r') as f:
                self.feature_names = json.load(f)
            print(f"✓ Loaded {len(self.feature_names)} feature names")
        
        # Load scaler if available
        if scaler_path and Path(scaler_path).exists():
            import pickle
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            print(f"✓ Loaded feature scaler")
        
        # Load model
        state_dict = torch.load(model_path, map_location=self.device)
        input_dim = state_dict['fc1.weight'].shape[1]
        
        self.model = NoCClassifier(input_dim=input_dim)
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"✓ Loaded model from {model_path}")
        print(f"  Input dimension: {input_dim}")
        print(f"  Device: {self.device}")
    
    def predict(self, features: Union[np.ndarray, List]) -> np.ndarray:
        """
        Predict NoC for samples.
        
        Args:
            features: np.ndarray of shape (n_samples, n_features) or single sample
        
        Returns:
            np.ndarray of predicted NoC (1-10)
        """
        # Convert to numpy if needed
        if isinstance(features, list):
            features = np.array(features)
        
        # Handle single sample
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Normalize if scaler available
        if self.scaler is not None:
            features = self.scaler.transform(features)
        
        # Convert to tensor
        X = torch.from_numpy(features).float().to(self.device)
        
        # Predict
        with torch.no_grad():
            logits = self.model(X)
            preds = torch.argmax(logits, dim=1) + 1  # NoC: 1-10
        
        return preds.cpu().numpy()
    
    def predict_proba(self, features: Union[np.ndarray, List]) -> np.ndarray:
        """
        Get probability distribution for each class.
        
        Args:
            features: np.ndarray of shape (n_samples, n_features)
        
        Returns:
            np.ndarray of shape (n_samples, 10) with probabilities for NoC 1-10
        """
        # Convert to numpy if needed
        if isinstance(features, list):
            features = np.array(features)
        
        # Handle single sample
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Normalize if scaler available
        if self.scaler is not None:
            features = self.scaler.transform(features)
        
        # Convert to tensor
        X = torch.from_numpy(features).float().to(self.device)
        
        # Get probabilities
        probs = self.model.predict_proba(X)
        
        return probs.cpu().numpy()
    
    def predict_with_confidence(self, features: Union[np.ndarray, List]) -> pd.DataFrame:
        """
        Predict NoC with confidence scores.
        
        Args:
            features: np.ndarray of shape (n_samples, n_features)
        
        Returns:
            DataFrame with columns: sample_id, predicted_noc, confidence
        """
        preds = self.predict(features)
        probs = self.predict_proba(features)
        confidences = np.max(probs, axis=1)
        
        df = pd.DataFrame({
            'sample_id': range(len(preds)),
            'predicted_noc': preds,
            'confidence': confidences
        })
        
        return df


# ============================================================================
# Example Usage
# ============================================================================

def example_usage():
    """
    Example showing how to use the predictor in a Kaggle notebook.
    """
    
    print("=" * 80)
    print("Example: Using Trained NoC Classifier for Predictions")
    print("=" * 80)
    
    # Paths (adjust for your Kaggle environment)
    model_path = "/kaggle/output/model.pt"
    feature_names_path = "/kaggle/output/feature_names.json"
    
    # Initialize predictor
    predictor = NoCPredictor(
        model_path=model_path,
        feature_names_path=feature_names_path
    )
    
    # Example 1: Predict single sample
    print("\n📌 Example 1: Single Sample Prediction")
    sample_features = np.random.randn(1, 78)  # Replace with real features
    pred = predictor.predict(sample_features)
    print(f"  Predicted NoC: {pred[0]}")
    
    # Example 2: Batch predictions with confidence scores
    print("\n📌 Example 2: Batch Predictions with Confidence")
    batch_features = np.random.randn(5, 78)  # 5 samples
    results_df = predictor.predict_with_confidence(batch_features)
    print(results_df)
    
    # Example 3: Get probability distribution
    print("\n📌 Example 3: Probability Distribution (per-class)")
    sample_features = np.random.randn(1, 78)
    probs = predictor.predict_proba(sample_features)
    
    for noc in range(1, 11):
        print(f"  P(NoC={noc}): {probs[0, noc-1]:.4f}")
    
    print(f"\n✅ Prediction examples complete!")


if __name__ == '__main__':
    example_usage()
