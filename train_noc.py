"""
NoC (Number of Contributors) Training Script

Trains the NoC classifier on mixture DNA electropherograms.

Usage:
    python train_noc.py \\
        --noc-labels data/noc_labels.csv \\
        --data-dir data/mixture_hid_files \\
        --output-dir output/noc_model \\
        --epochs 50 \\
        --batch-size 32
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

import mlflow
import numpy as np
import torch
from sklearn.model_selection import train_test_split

from config_io import load_config
from DNAnet.data.noc_dataset import NoCDataset
from DNAnet.models.noc_classifier import NoCClassifier
from DNAnet.evaluation.noc_metrics import compute_noc_metrics
from utils import add_file_handler_to_logger, prepare_output_file


LOGGER = logging.getLogger('dnanet_noc')


def run(
    noc_labels: str,
    data_dir: str,
    output_dir: Optional[str] = None,
    panel_config: Optional[str] = None,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    val_split: float = 0.2,
    early_stopping_patience: int = 10,
    quiet: bool = False,
    use_mlflow: bool = False,
    seed: int = 42,
):
    """
    Train NoC classifier.
    
    Args:
        noc_labels: Path to CSV with NoC labels (columns: filename, noc)
        data_dir: Directory containing mixture HID files
        output_dir: Output directory for results
        panel_config: Path to panel config file (optional)
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate for optimizer
        val_split: Validation split ratio
        early_stopping_patience: Early stopping patience
        quiet: Suppress verbose logging
        use_mlflow: Whether to use MLflow tracking
        seed: Random seed
    """
    # Setup
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if not output_dir:
        output_dir = f'output/train_noc_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_level = logging.WARNING if quiet else logging.INFO
    logging.basicConfig(level=log_level)
    
    log_path = output_dir / 'log_training.txt'
    add_file_handler_to_logger(LOGGER, path=str(log_path), level=log_level)
    LOGGER.info(f"Logs will be written to {log_path}")
    
    # Setup MLflow
    if use_mlflow:
        LOGGER.info("Configuring MLflow")
        mlflow.start_run(run_name=f'noc_train_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        mlflow.log_params({
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'val_split': val_split,
            'seed': seed,
        })
    
    # Load data
    LOGGER.info(f"Loading mixture dataset from {data_dir}")
    dataset = NoCDataset(
        root_dir=data_dir,
        noc_labels_path=noc_labels,
        use_cache=False,
        feature_normalize=True,
    )
    
    LOGGER.info(f"Loaded {len(dataset)} samples")
    
    # Compute feature statistics on full dataset for normalization
    LOGGER.info("Computing feature statistics for normalization...")
    dataset.compute_feature_statistics()
    feature_names = dataset.get_feature_names()
    num_features = len(feature_names)
    LOGGER.info(f"Total features: {num_features}")
    
    # Extract all features and labels
    LOGGER.info("Extracting features...")
    all_features = []
    all_labels = []
    for i in range(len(dataset)):
        try:
            item = dataset[i]
            all_features.append(item['features'])
            all_labels.append(item['noc'])
        except Exception as e:
            LOGGER.warning(f"Skipping sample {i}: {e}")
            continue
    
    all_features = np.array(all_features)
    all_labels = np.array(all_labels)
    
    LOGGER.info(f"Extracted {len(all_labels)} valid samples")
    LOGGER.info(f"NoC distribution: {np.bincount(all_labels)}")
    
    # Train/val split (stratified by NoC)
    LOGGER.info(f"Splitting data: {100*(1-val_split)}% train, {100*val_split}% val")
    train_idx, val_idx = train_test_split(
        np.arange(len(all_labels)),
        test_size=val_split,
        stratify=all_labels,
        random_state=seed
    )
    
    train_features = all_features[train_idx]
    train_labels = all_labels[train_idx]
    val_features = all_features[val_idx]
    val_labels = all_labels[val_idx]
    
    LOGGER.info(f"Train set: {len(train_labels)} samples")
    LOGGER.info(f"Val set: {len(val_labels)} samples")
    
    # Initialize model
    LOGGER.info("Initializing model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    LOGGER.info(f"Using device: {device}")
    
    model = NoCClassifier(
        num_features=num_features,
        num_classes=10,  # NoC from 1-10 contributors
        hidden_dim=128,
        dropout_rate=0.2,
        device=device,
    )
    
    # Train
    LOGGER.info("Starting training...")
    training_history = model.fit(
        train_features=train_features,
        train_labels=train_labels,
        val_features=val_features,
        val_labels=val_labels,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        early_stopping_patience=early_stopping_patience,
    )
    
    # Evaluate on validation set
    LOGGER.info("Evaluating on validation set...")
    val_preds = model.predict_batch(val_features)
    val_metrics = compute_noc_metrics(val_labels, val_preds, return_per_class=True)
    
    LOGGER.info(f"Validation Accuracy: {val_metrics['accuracy']:.4f}")
    LOGGER.info(f"Validation Balanced Accuracy: {val_metrics['balanced_accuracy']:.4f}")
    LOGGER.info(f"Validation F1 (macro): {val_metrics['f1_macro']:.4f}")
    LOGGER.info(f"Validation MAE: {val_metrics['mae']:.4f}")
    LOGGER.info(f"Off-by-one error: {val_metrics['off_by_one_error']:.4f}")
    
    # Log metrics to MLflow
    if use_mlflow:
        mlflow.log_metrics({
            'val_accuracy': val_metrics['accuracy'],
            'val_balanced_accuracy': val_metrics['balanced_accuracy'],
            'val_f1_macro': val_metrics['f1_macro'],
            'val_mae': val_metrics['mae'],
        })
    
    # Also evaluate on training set
    LOGGER.info("Evaluating on training set...")
    train_preds = model.predict_batch(train_features)
    train_metrics = compute_noc_metrics(train_labels, train_preds, return_per_class=False)
    LOGGER.info(f"Training Accuracy: {train_metrics['accuracy']:.4f}")
    
    # Save model
    LOGGER.info(f"Saving model to {output_dir}")
    model.save(output_dir)
    
    # Save results
    results = {
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'validation_per_class': val_metrics.get('per_class', {}),
        'training_history': {
            'train_loss': [float(x) for x in training_history['train_loss']],
            'val_loss': [float(x) for x in training_history['val_loss']],
            'train_acc': [float(x) for x in training_history['train_acc']],
            'val_acc': [float(x) for x in training_history['val_acc']],
        },
        'hyperparameters': {
            'num_features': num_features,
            'num_classes': 10,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'epochs': epochs,
            'val_split': val_split,
        }
    }
    
    results_path = output_dir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    LOGGER.info(f"Results saved to {results_path}")
    
    # Save feature names for inference
    feature_names_path = output_dir / 'feature_names.json'
    with open(feature_names_path, 'w') as f:
        json.dump(feature_names, f, indent=2)
    
    # Save feature statistics
    feature_stats_path = output_dir / 'feature_stats.json'
    with open(feature_stats_path, 'w') as f:
        json.dump(model.training_history, f, indent=2, default=str)
    
    if use_mlflow:
        mlflow.end_run()
    
    LOGGER.info("Training completed!")
    return results


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Train NoC classifier for DNA mixture prediction'
    )
    parser.add_argument(
        '--noc-labels',
        required=True,
        help='Path to CSV with NoC labels (columns: filename, noc)'
    )
    parser.add_argument(
        '--data-dir',
        required=True,
        help='Directory containing mixture HID files'
    )
    parser.add_argument(
        '--output-dir',
        help='Output directory (default: output/train_noc_TIMESTAMP)'
    )
    parser.add_argument(
        '--panel-config',
        help='Path to panel config file (optional)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs (default: 50)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size (default: 32)'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=1e-3,
        help='Learning rate (default: 1e-3)'
    )
    parser.add_argument(
        '--val-split',
        type=float,
        default=0.2,
        help='Validation split ratio (default: 0.2)'
    )
    parser.add_argument(
        '--early-stopping-patience',
        type=int,
        default=10,
        help='Early stopping patience (default: 10)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    parser.add_argument(
        '--use-mlflow',
        action='store_true',
        help='Use MLflow for experiment tracking'
    )
    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Suppress verbose logging'
    )
    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    
    run(
        noc_labels=args.noc_labels,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        panel_config=args.panel_config,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        val_split=args.val_split,
        early_stopping_patience=args.early_stopping_patience,
        quiet=args.quiet,
        use_mlflow=args.use_mlflow,
        seed=args.seed,
    )
