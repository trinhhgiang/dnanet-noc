"""
NoC (Number of Contributors) Classifier Model.

Predicts number of contributors from mixture DNA features using a neural network.
"""

from typing import Dict, Optional, Sequence, Tuple, Any
from pathlib import Path
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from loguru import logger

from DNAnet.models.base import TrainableModel, TORCH_DEFAULT_DEVICE
from DNAnet.models.prediction import Prediction


class NoCClassifier(TrainableModel):
    """
    Fully-connected neural network for NoC prediction.
    
    Architecture:
    Input (features) -> FC(128) -> ReLU -> Dropout -> FC(64) -> ReLU -> FC(num_classes)
    """
    
    def __init__(
        self,
        num_features: int = 100,
        num_classes: int = 10,
        hidden_dim: int = 128,
        dropout_rate: float = 0.2,
        device: Optional[str] = None,
    ):
        """
        Args:
            num_features: Number of input features
            num_classes: Number of NoC classes (1-10 contributors)
            hidden_dim: Hidden layer dimension
            dropout_rate: Dropout probability
            device: 'cuda' or 'cpu'
        """
        super().__init__()
        
        self.num_features = num_features
        self.num_classes = num_classes
        self.device = device or TORCH_DEFAULT_DEVICE
        
        # Build network
        self.network = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )
        
        self.to(self.device)
        self.loss_fn = nn.CrossEntropyLoss()
        
        # Training state
        self.optimizer = None
        self.best_val_loss = float('inf')
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
    
    def predict(self, features: np.ndarray) -> int:
        """
        Predict NoC for a single feature vector.
        
        Args:
            features: np.ndarray of shape (num_features,)
        
        Returns:
            Predicted NoC (1-indexed, 1-10)
        """
        self.eval()
        with torch.no_grad():
            features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
            logits = self.network(features_tensor)
            pred_class = torch.argmax(logits, dim=1).item()
        
        return pred_class + 1  # Convert 0-indexed to 1-indexed
    
    def predict_batch(self, features_list: Sequence[np.ndarray]) -> np.ndarray:
        """
        Predict NoC for multiple feature vectors.
        
        Args:
            features_list: Sequence of feature arrays
        
        Returns:
            np.ndarray of predicted NoCs (1-indexed)
        """
        self.eval()
        with torch.no_grad():
            features_tensor = torch.tensor(
                np.array(features_list),
                dtype=torch.float32
            ).to(self.device)
            logits = self.network(features_tensor)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
        
        return preds + 1  # Convert to 1-indexed
    
    def predict_proba(self, features_list: Sequence[np.ndarray]) -> np.ndarray:
        """
        Get probability distribution over NoC classes.
        
        Args:
            features_list: Sequence of feature arrays
        
        Returns:
            np.ndarray of shape (num_samples, num_classes) with probabilities
        """
        self.eval()
        with torch.no_grad():
            features_tensor = torch.tensor(
                np.array(features_list),
                dtype=torch.float32
            ).to(self.device)
            logits = self.network(features_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        
        return probs
    
    def fit(
        self,
        train_features: np.ndarray,
        train_labels: np.ndarray,
        val_features: Optional[np.ndarray] = None,
        val_labels: Optional[np.ndarray] = None,
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        early_stopping_patience: int = 10,
        min_delta: float = 1e-4,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train the classifier.
        
        Args:
            train_features: np.ndarray of shape (num_samples, num_features)
            train_labels: np.ndarray of shape (num_samples,) with labels 1-10
            val_features: Validation features (optional)
            val_labels: Validation labels (optional)
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
            weight_decay: L2 regularization
            early_stopping_patience: Patience for early stopping
            min_delta: Minimum improvement threshold
        
        Returns:
            Dictionary with training history
        """
        # Setup optimizer
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Convert labels to 0-indexed for PyTorch
        train_labels_zero = train_labels - 1
        
        # Create training dataset
        train_dataset = TensorDataset(
            torch.tensor(train_features, dtype=torch.float32),
            torch.tensor(train_labels_zero, dtype=torch.long)
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Create validation dataset if provided
        if val_features is not None and val_labels is not None:
            val_labels_zero = val_labels - 1
            val_dataset = TensorDataset(
                torch.tensor(val_features, dtype=torch.float32),
                torch.tensor(val_labels_zero, dtype=torch.long)
            )
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        else:
            val_loader = None
        
        # Training loop
        patience_counter = 0
        
        for epoch in range(epochs):
            # Train epoch
            train_loss, train_acc = self._train_epoch(train_loader)
            self.training_history['train_loss'].append(train_loss)
            self.training_history['train_acc'].append(train_acc)
            
            # Validation epoch
            if val_loader is not None:
                val_loss, val_acc = self._validate_epoch(val_loader)
                self.training_history['val_loss'].append(val_loss)
                self.training_history['val_acc'].append(val_acc)
                
                logger.info(
                    f"Epoch {epoch+1}/{epochs} | "
                    f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                    f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}"
                )
                
                # Early stopping
                if val_loss < self.best_val_loss - min_delta:
                    self.best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        logger.info(f"Early stopping at epoch {epoch+1}")
                        break
            else:
                logger.info(
                    f"Epoch {epoch+1}/{epochs} | "
                    f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}"
                )
        
        return self.training_history
    
    def _train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Run one training epoch."""
        self.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for features, labels in train_loader:
            features = features.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward
            logits = self.network(features)
            loss = self.loss_fn(logits, labels)
            
            # Backward
            loss.backward()
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item() * features.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        
        avg_loss = total_loss / total
        accuracy = correct / total
        return avg_loss, accuracy
    
    def _validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Run one validation epoch."""
        self.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                logits = self.network(features)
                loss = self.loss_fn(logits, labels)
                
                total_loss += loss.item() * features.size(0)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        avg_loss = total_loss / total
        accuracy = correct / total
        return avg_loss, accuracy
    
    def save(self, model_dir: Path) -> None:
        """Save model weights and config."""
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save weights
        weights_path = model_dir / 'noc_classifier_weights.pt'
        torch.save(self.network.state_dict(), weights_path)
        
        # Save config
        config = {
            'num_features': self.num_features,
            'num_classes': self.num_classes,
        }
        config_path = model_dir / 'noc_classifier_config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Model saved to {model_dir}")
    
    def load(self, model_dir: Path) -> None:
        """Load model weights and config."""
        model_dir = Path(model_dir)
        
        # Load config
        config_path = model_dir / 'noc_classifier_config.json'
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            self.num_features = config['num_features']
            self.num_classes = config['num_classes']
        
        # Load weights
        weights_path = model_dir / 'noc_classifier_weights.pt'
        if weights_path.exists():
            self.network.load_state_dict(torch.load(weights_path, map_location=self.device))
            logger.info(f"Model loaded from {model_dir}")
        else:
            raise FileNotFoundError(f"Model weights not found at {weights_path}")
