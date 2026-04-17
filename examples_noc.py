"""
Example: End-to-end NoC prediction workflow

This script demonstrates:
1. Loading mixture data
2. Training NoC classifier
3. Evaluating on test set
4. Making predictions on new samples
"""

import numpy as np
from pathlib import Path

from DNAnet.data.noc_dataset import NoCDataset
from DNAnet.models.noc_classifier import NoCClassifier
from DNAnet.evaluation.noc_metrics import compute_noc_metrics


def example_basic_inference():
    """
    Example 1: Load trained model and predict on single sample
    """
    print("=" * 60)
    print("Example 1: Basic Inference")
    print("=" * 60)
    
    # Load trained model
    model_dir = Path('output/my_noc_model')
    model = NoCClassifier(num_features=100, num_classes=10)
    model.load(model_dir)
    
    # Load test dataset
    dataset = NoCDataset(
        root_dir='data/test_samples',
        noc_labels_path='data/test_labels.csv',
        use_cache=False,
        feature_normalize=True,
    )
    
    # Get first sample
    item = dataset[0]
    features = item['features']
    true_noc = item['noc']
    
    # Predict
    pred_noc = model.predict(features)
    pred_proba = model.predict_proba([features])[0]
    
    print(f"File: {item['metadata']['filename']}")
    print(f"True NoC: {true_noc}")
    print(f"Predicted NoC: {pred_noc}")
    print(f"Confidence: {pred_proba[pred_noc-1]:.2%}")
    print(f"All probabilities: {dict(enumerate(pred_proba, 1))}")
    print()


def example_batch_prediction():
    """
    Example 2: Batch prediction on multiple samples
    """
    print("=" * 60)
    print("Example 2: Batch Prediction")
    print("=" * 60)
    
    # Load model
    model_dir = Path('output/my_noc_model')
    model = NoCClassifier(num_features=100, num_classes=10)
    model.load(model_dir)
    
    # Load test dataset
    dataset = NoCDataset(
        root_dir='data/test_samples',
        noc_labels_path='data/test_labels.csv',
        use_cache=False,
        feature_normalize=True,
    )
    
    # Get first 10 samples
    features_list = []
    true_labels = []
    filenames = []
    
    for i in range(min(10, len(dataset))):
        item = dataset[i]
        features_list.append(item['features'])
        true_labels.append(item['noc'])
        filenames.append(item['metadata']['filename'])
    
    features_array = np.array(features_list)
    
    # Batch predict
    preds = model.predict_batch(features_array)
    probs = model.predict_proba(features_array)
    
    # Print results
    print(f"{'File':<30} {'True':<6} {'Pred':<6} {'Confident':<10}")
    print("-" * 52)
    for fname, true, pred, prob in zip(filenames, true_labels, preds, probs):
        confidence = prob[pred-1]
        symbol = '✓' if true == pred else '✗'
        print(f"{fname:<30} {true:<6} {pred:<6} {symbol} {confidence:>7.2%}")
    print()


def example_evaluation():
    """
    Example 3: Evaluate model on full test set
    """
    print("=" * 60)
    print("Example 3: Full Evaluation")
    print("=" * 60)
    
    # Load model
    model_dir = Path('output/my_noc_model')
    model = NoCClassifier(num_features=100, num_classes=10)
    model.load(model_dir)
    
    # Load test dataset
    dataset = NoCDataset(
        root_dir='data/test_samples',
        noc_labels_path='data/test_labels.csv',
        use_cache=False,
        feature_normalize=True,
    )
    
    # Extract all features and labels
    all_features = []
    all_labels = []
    
    for i in range(len(dataset)):
        item = dataset[i]
        all_features.append(item['features'])
        all_labels.append(item['noc'])
    
    all_features = np.array(all_features)
    all_labels = np.array(all_labels)
    
    # Predict
    preds = model.predict_batch(all_features)
    
    # Compute metrics
    metrics = compute_noc_metrics(all_labels, preds, return_per_class=True)
    
    print(f"Test Set Size: {len(all_labels)} samples")
    print(f"\nOverall Metrics:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"  Precision (macro): {metrics['precision_macro']:.4f}")
    print(f"  Recall (macro): {metrics['recall_macro']:.4f}")
    print(f"  F1-score (macro): {metrics['f1_macro']:.4f}")
    print(f"  MAE: {metrics['mae']:.4f}")
    print(f"  Off-by-one error: {metrics['off_by_one_error']:.2%}")
    
    print(f"\nPer-Class Metrics:")
    for noc, class_metrics in sorted(metrics['per_class'].items()):
        print(f"  NoC={noc}:")
        print(f"    Precision={class_metrics['precision']:.3f}, "
              f"Recall={class_metrics['recall']:.3f}, "
              f"F1={class_metrics['f1_score']:.3f}, "
              f"Support={class_metrics['support']}")
    print()


def example_feature_inspection():
    """
    Example 4: Inspect features extracted from a sample
    """
    print("=" * 60)
    print("Example 4: Feature Inspection")
    print("=" * 60)
    
    # Load dataset
    dataset = NoCDataset(
        root_dir='data/test_samples',
        noc_labels_path='data/test_labels.csv',
        use_cache=False,
        feature_normalize=False,  # Don't normalize to see raw values
    )
    
    # Get feature names
    feature_names = dataset.get_feature_names()
    print(f"Total features: {len(feature_names)}")
    print(f"\nFirst 20 feature names:")
    for i, name in enumerate(feature_names[:20]):
        print(f"  {i+1}. {name}")
    print(f"  ...")
    
    # Get a sample and show its features
    item = dataset[0]
    features = item['features']
    true_noc = item['noc']
    
    print(f"\nSample features (NoC={true_noc}):")
    print(f"{'Feature':<40} Value")
    print("-" * 50)
    for name, value in zip(feature_names[:15], features[:15]):
        print(f"{name:<40} {value:>8.2f}")
    print("  ...")
    print()


def example_training_from_scratch():
    """
    Example 5: Train a model from scratch
    
    Note: This requires actual data; see train_noc.py for full training script
    """
    print("=" * 60)
    print("Example 5: Training from Scratch")
    print("=" * 60)
    
    print("This example shows programmatic training (see train_noc.py for full script)")
    print()
    
    # Load dataset
    dataset = NoCDataset(
        root_dir='data/mixture_samples',
        noc_labels_path='data/mixture_labels.csv',
        use_cache=False,
        feature_normalize=True,
    )
    
    print(f"Loaded dataset with {len(dataset)} samples")
    
    # Compute feature statistics for normalization
    print("Computing feature statistics...")
    dataset.compute_feature_statistics()
    feature_names = dataset.get_feature_names()
    num_features = len(feature_names)
    print(f"  {num_features} features")
    
    # Extract features and labels (simplified)
    all_features = []
    all_labels = []
    for i in range(min(100, len(dataset))):  # Use first 100 for speed
        item = dataset[i]
        all_features.append(item['features'])
        all_labels.append(item['noc'])
    
    all_features = np.array(all_features)
    all_labels = np.array(all_labels)
    
    print(f"Extracted {len(all_labels)} training samples")
    print(f"  NoC distribution: {np.bincount(all_labels)}")
    
    # Initialize model
    model = NoCClassifier(
        num_features=num_features,
        num_classes=10,
        hidden_dim=128,
        dropout_rate=0.2,
    )
    
    # Train (simplified; no validation split for this example)
    print("\nTraining model...")
    history = model.fit(
        train_features=all_features,
        train_labels=all_labels,
        epochs=5,  # Just 5 epochs for example
        batch_size=16,
        learning_rate=1e-3,
    )
    
    print(f"✓ Training complete")
    print(f"  Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"  Final train acc: {history['train_acc'][-1]:.4f}")
    print()


def main():
    """Run all examples"""
    print("\n" + "="*60)
    print("DNANet NoC Prediction Examples")
    print("="*60 + "\n")
    
    try:
        example_basic_inference()
    except FileNotFoundError as e:
        print(f"⚠️  Skipping inference example: {e}\n")
    
    try:
        example_batch_prediction()
    except FileNotFoundError as e:
        print(f"⚠️  Skipping batch prediction example: {e}\n")
    
    try:
        example_evaluation()
    except FileNotFoundError as e:
        print(f"⚠️  Skipping evaluation example: {e}\n")
    
    try:
        example_feature_inspection()
    except FileNotFoundError as e:
        print(f"⚠️  Skipping feature inspection example: {e}\n")
    
    try:
        example_training_from_scratch()
    except FileNotFoundError as e:
        print(f"⚠️  Skipping training example: {e}\n")
    
    print("=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
