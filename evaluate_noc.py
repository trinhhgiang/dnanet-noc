"""
NoC Evaluation Script

Evaluates a trained NoC classifier on mixture DNA samples.

Usage:
    python evaluate_noc.py \\
        --model-dir output/noc_model \\
        --test-data data/test_mixture_hid_files \\
        --noc-labels data/test_noc_labels.csv \\
        --output-dir output/noc_evaluation
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict

import numpy as np
import pandas as pd
import torch

from DNAnet.data.noc_dataset import NoCDataset
from DNAnet.models.noc_classifier import NoCClassifier
from DNAnet.evaluation.noc_metrics import (
    compute_noc_metrics,
    noc_confusion_matrix,
    noc_confusion_matrix_normalized,
)
from utils import add_file_handler_to_logger, prepare_output_file


LOGGER = logging.getLogger('dnanet_noc_eval')


def run(
    model_dir: str,
    test_data: str,
    noc_labels: str,
    output_dir: Optional[str] = None,
    batch_size: int = 32,
    quiet: bool = False,
):
    """
    Evaluate NoC classifier on test set.
    
    Args:
        model_dir: Path to trained model directory
        test_data: Path to test HID files directory
        noc_labels: Path to test NoC labels CSV
        output_dir: Output directory for results
        batch_size: Batch size for inference
        quiet: Suppress verbose logging
    """
    # Setup
    if not output_dir:
        output_dir = f'output/eval_noc_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_level = logging.WARNING if quiet else logging.INFO
    logging.basicConfig(level=log_level)
    
    log_path = output_dir / 'log_evaluation.txt'
    add_file_handler_to_logger(LOGGER, path=str(log_path), level=log_level)
    LOGGER.info(f"Logs will be written to {log_path}")
    
    # Load model
    LOGGER.info(f"Loading model from {model_dir}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    LOGGER.info(f"Using device: {device}")
    
    # Load config to determine num_features
    config_path = Path(model_dir) / 'noc_classifier_config.json'
    with open(config_path) as f:
        config = json.load(f)
    
    model = NoCClassifier(
        num_features=config['num_features'],
        num_classes=config['num_classes'],
        device=device,
    )
    model.load(Path(model_dir))
    
    # Load test dataset
    LOGGER.info(f"Loading test dataset from {test_data}")
    test_dataset = NoCDataset(
        root_dir=test_data,
        noc_labels_path=noc_labels,
        use_cache=False,
        feature_normalize=True,
    )
    
    # Load feature statistics from training (saved during training)
    feature_stats_path = Path(model_dir) / 'feature_stats.json'
    if feature_stats_path.exists():
        LOGGER.info("Loading feature statistics for normalization...")
        with open(feature_stats_path) as f:
            stats_data = json.load(f)
        # Would need to properly deserialize; for now, let test_dataset compute its own
    
    LOGGER.info(f"Test set size: {len(test_dataset)}")
    
    # Extract all features and labels
    LOGGER.info("Extracting test features...")
    all_features = []
    all_labels = []
    all_metadata = []
    
    for i in range(len(test_dataset)):
        try:
            item = test_dataset[i]
            all_features.append(item['features'])
            all_labels.append(item['noc'])
            all_metadata.append(item['metadata'])
        except Exception as e:
            LOGGER.warning(f"Skipping sample {i}: {e}")
            continue
    
    all_features = np.array(all_features)
    all_labels = np.array(all_labels)
    
    LOGGER.info(f"Extracted {len(all_labels)} valid test samples")
    LOGGER.info(f"NoC distribution in test set: {np.bincount(all_labels)}")
    
    # Run inference
    LOGGER.info("Running inference...")
    test_preds = model.predict_batch(all_features)
    test_probs = model.predict_proba(all_features)
    
    # Compute metrics
    LOGGER.info("Computing metrics...")
    metrics = compute_noc_metrics(all_labels, test_preds, return_per_class=True)
    
    LOGGER.info(f"Test Accuracy: {metrics['accuracy']:.4f}")
    LOGGER.info(f"Test Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    LOGGER.info(f"Test Precision (macro): {metrics['precision_macro']:.4f}")
    LOGGER.info(f"Test Recall (macro): {metrics['recall_macro']:.4f}")
    LOGGER.info(f"Test F1 (macro): {metrics['f1_macro']:.4f}")
    LOGGER.info(f"Test F1 (weighted): {metrics['f1_weighted']:.4f}")
    LOGGER.info(f"Mean Absolute Error: {metrics['mae']:.4f}")
    LOGGER.info(f"Off-by-one error: {metrics['off_by_one_error']:.4f}")
    
    # Per-class metrics
    LOGGER.info("\nPer-class metrics:")
    for noc, class_metrics in sorted(metrics['per_class'].items()):
        LOGGER.info(
            f"NoC={noc}: Precision={class_metrics['precision']:.4f}, "
            f"Recall={class_metrics['recall']:.4f}, "
            f"F1={class_metrics['f1_score']:.4f}, "
            f"Support={class_metrics['support']}"
        )
    
    # Confusion matrix
    cm = noc_confusion_matrix(all_labels, test_preds)
    cm_norm = noc_confusion_matrix_normalized(all_labels, test_preds)
    
    # Create detailed results dataframe
    results_df = pd.DataFrame({
        'filename': [m.get('filename', '') for m in all_metadata],
        'true_noc': all_labels,
        'predicted_noc': test_preds,
        'difference': all_labels - test_preds,
    })
    
    # Add probabilities for each class
    for c in range(1, 11):
        results_df[f'prob_noc_{c}'] = test_probs[:, c-1]
    
    # Add prediction confidence (max probability)
    results_df['confidence'] = np.max(test_probs, axis=1)
    
    # Save results
    LOGGER.info(f"Saving results to {output_dir}")
    
    # Save detailed predictions
    predictions_path = output_dir / 'predictions.csv'
    results_df.to_csv(predictions_path, index=False)
    LOGGER.info(f"Predictions saved to {predictions_path}")
    
    # Save metrics
    metrics_path = output_dir / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    LOGGER.info(f"Metrics saved to {metrics_path}")
    
    # Save confusion matrix
    cm_path = output_dir / 'confusion_matrix.json'
    with open(cm_path, 'w') as f:
        json.dump({
            'raw': cm.tolist(),
            'normalized': cm_norm.tolist(),
            'labels': list(range(1, 11)),
        }, f, indent=2)
    LOGGER.info(f"Confusion matrix saved to {cm_path}")
    
    # Save summary report
    summary = {
        'test_set': str(test_data),
        'model_dir': str(model_dir),
        'n_samples': len(all_labels),
        'metrics': {
            'accuracy': float(metrics['accuracy']),
            'balanced_accuracy': float(metrics['balanced_accuracy']),
            'precision_macro': float(metrics['precision_macro']),
            'recall_macro': float(metrics['recall_macro']),
            'f1_macro': float(metrics['f1_macro']),
            'f1_weighted': float(metrics['f1_weighted']),
            'mae': float(metrics['mae']),
            'off_by_one_error': float(metrics['off_by_one_error']),
        },
        'per_class_metrics': metrics['per_class'],
    }
    
    summary_path = output_dir / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    LOGGER.info(f"Summary saved to {summary_path}")
    
    # Analysis: Misclassified samples
    misclassified = results_df[results_df['true_noc'] != results_df['predicted_noc']]
    LOGGER.info(f"\nMisclassified samples: {len(misclassified)} / {len(results_df)}")
    
    if len(misclassified) > 0:
        misclass_path = output_dir / 'misclassified.csv'
        misclassified.to_csv(misclass_path, index=False)
        LOGGER.info(f"Misclassified samples saved to {misclass_path}")
        
        # Analyze misclassification patterns
        LOGGER.info("Misclassification patterns:")
        for true_noc in sorted(results_df['true_noc'].unique()):
            subset = misclassified[misclassified['true_noc'] == true_noc]
            if len(subset) > 0:
                pred_dist = subset['predicted_noc'].value_counts().sort_index()
                LOGGER.info(f"  True NoC={true_noc} misclassified as: {dict(pred_dist)}")
    
    LOGGER.info("\nEvaluation completed!")
    return metrics


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Evaluate NoC classifier on test set'
    )
    parser.add_argument(
        '--model-dir',
        required=True,
        help='Path to trained model directory'
    )
    parser.add_argument(
        '--test-data',
        required=True,
        help='Path to test HID files directory'
    )
    parser.add_argument(
        '--noc-labels',
        required=True,
        help='Path to test set NoC labels CSV'
    )
    parser.add_argument(
        '--output-dir',
        help='Output directory (default: output/eval_noc_TIMESTAMP)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for inference (default: 32)'
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
        model_dir=args.model_dir,
        test_data=args.test_data,
        noc_labels=args.noc_labels,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        quiet=args.quiet,
    )
