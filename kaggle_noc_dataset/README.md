# DNANet NoC (Number of Contributors) Prediction Module

This module extends DNANet to predict **Number of Contributors (NoC)** in DNA mixtures using machine learning classification.

## Overview

The NoC module provides:
- **Mixture-specific feature extraction** (~100 features per electropherogram)
- **Fully-connected classifier** for predicting contributor count (1-10)
- **Comprehensive evaluation metrics** (accuracy, F1-score, confusion matrix, per-class metrics)
- **End-to-end training and evaluation pipelines**

## Architecture

```
Mixture HID File
    ↓
[parse_raw_hid.py] → Extract peak data
    ↓
[baseline_and_smooth.py] → Baseline correction
    ↓
[mixture_features.py] → Extract ~100 statistical features
    ├─ Per-locus: peak count, heights, ratios, position spread
    ├─ Global: total peaks, RFU distribution (mean, std, skewness, entropy)
    └─ Anomaly: tetrallelic loci, peak density
    ↓
[NoCClassifier] → FC NN: features → softmax → NoC (1-10)
    ↓
[Predictions + Probabilities] → per-sample predicted contributor count
```

## Installation & Dependencies

Required packages (installed via `pyproject.toml`):
```
torch>=2.0.0
scikit-learn
pandas
numpy
scipy
```

## Quick Start

### 1. Prepare Data

Create a CSV file mapping filenames to ground-truth NoC:

**format: `mixture_labels.csv`**
```
filename,noc
mixture_001.hid,2
mixture_002.hid,3
mixture_003.hid,2
...
```

Organize .hid files in a directory:
```
data/mixture_samples/
├── mixture_001.hid
├── mixture_002.hid
├── mixture_003.hid
└── ...
```

### 2. Train Model

```bash
python train_noc.py \
    --noc-labels data/mixture_labels.csv \
    --data-dir data/mixture_samples \
    --output-dir output/my_noc_model \
    --epochs 50 \
    --batch-size 32 \
    --learning-rate 1e-3 \
    --val-split 0.2
```

**Output:**
- `noc_classifier_weights.pt` — Trained model weights
- `noc_classifier_config.json` — Model configuration
- `results.json` — Training metrics & history
- `feature_names.json` — Ordered feature names (for inference)
- `log_training.txt` — Training log

### 3. Evaluate Model

```bash
python evaluate_noc.py \
    --model-dir output/my_noc_model \
    --test-data data/test_mixture_samples \
    --noc-labels data/test_labels.csv \
    --output-dir output/evaluation_results
```

**Output:**
- `predictions.csv` — Per-sample predictions + probabilities
- `metrics.json` — Overall accuracy, F1-score, per-class metrics
- `confusion_matrix.json` — Confusion matrix (raw & normalized)
- `summary.json` — High-level summary
- `misclassified.csv` — Problematic samples (if any)

### 4. Use in Python Code

```python
from DNAnet.data.noc_dataset import NoCDataset
from DNAnet.models.noc_classifier import NoCClassifier
import numpy as np

# Load trained model
model = NoCClassifier(num_features=100, num_classes=10)
model.load('output/my_noc_model')

# Extract features from a mixture
dataset = NoCDataset('data/test_samples', 'data/test_labels.csv')
item = dataset[0]
features = item['features']  # np.array of shape (100,)

# Predict
pred_noc = model.predict(features)  # Returns 1-10
pred_proba = model.predict_proba([features])  # Shape: (1, 10), probabilities
print(f"Predicted NoC: {pred_noc}")
print(f"Confidence: {pred_proba[0, pred_noc-1]:.2%}")
```

## Features Extracted

### Per-Locus Features (×17 loci = 136 features)
- **Count**: `num_peaks` (peaks detected)
- **Heights**: `max_rfu`, `mean_rfu`, `min_rfu`, `rfu_std`
- **Ratios**: `height_ratio_12`, `height_ratio_23`, `height_ratio_34` (peak intensity ratios)
- **Density**: `peak_density` (peaks per allele)
- **Position**: `position_spread`, `position_stddev` (peak clustering)
- **Ranking**: `ranked_rfu_0` through `ranked_rfu_3` (sorted peak heights)

### Global Features (11 features)
- **Totals**: `total_peaks`, `total_peaks_normalized`
- **RFU Stats**: `global_mean_rfu`, `global_std_rfu`, `global_max_rfu`, `global_min_rfu`
- **Distribution**: `global_rfu_cv`, `global_rfu_skewness`, `global_rfu_kurtosis`, `global_rfu_entropy`
- **Quantiles**: `global_rfu_q25`, `global_rfu_q50`, `global_rfu_q75`

### Locus-Wise Anomalies (7 features)
- `n_loci_with_4_peaks` (tetrallelic loci)
- `n_loci_with_3_peaks`, `n_loci_with_2_peaks`
- `max_peaks_any_locus`, `mean_peaks_per_locus`, `std_peaks_per_locus`
- `mean_rfu_across_loci_std` (RFU homogeneity)

**Total: ~100 features per electropherogram**

## Model Architecture

**NoCClassifier**
```
Input (100 features)
  ↓
Linear(100 → 128) + ReLU + Dropout(0.2)
  ↓
Linear(128 → 64) + ReLU + Dropout(0.2)
  ↓
Linear(64 → 10)  [Logits for classes 1-10]
  ↓
Cross-Entropy Loss (training) / Softmax (inference)
```

**Training Hyperparameters** (configurable):
- Optimizer: Adam (default lr=1e-3, weight_decay=1e-4)
- Loss: CrossEntropyLoss
- Early stopping: patience=10, min_delta=1e-4
- Batch size: 32 (default)
- Epochs: 50 (default)

## Evaluation Metrics

### Overall Metrics
- **Accuracy**: Fraction of correct predictions
- **Balanced Accuracy**: Macro-average per-class recall (good for imbalanced classes)
- **Precision/Recall/F1 (Macro)**: Unweighted average across classes
- **Precision/Recall/F1 (Weighted)**: Weighted by class support
- **MAE**: Mean absolute error in predicted NoC
- **Off-by-one error**: Fraction of predictions off by exactly ±1 contributor

### Per-Class Metrics
For each NoC (1-10), reports:
- Precision, Recall, F1-score
- Support (# samples)

### Confusion Matrix
- **Raw**: Count of predictions per true class
- **Normalized**: Row-normalized (shows where predictions go)

## Configuration

### Training Config: `config/training/noc_classification.yaml`
```yaml
training:
  epochs: 50
  batch_size: 32
  learning_rate: 0.001
  early_stopping_patience: 10
  validation_split: 0.2

model:
  hidden_dim: 128
  dropout_rate: 0.2
```

### Data Config: `config/data/noc_mixture.yaml`
```yaml
data:
  root_dir: "data/mixture_samples"
  noc_labels: "data/mixture_labels.csv"
  
features:
  normalize: true
  peak_threshold: 50.0
  n_top_loci: 17
```

## Known Limitations

1. **Biological Confounds**
   - Allelic dropout makes low-template mixtures appear to have fewer contributors
   - Stutter peaks can mimic additional alleles
   - Population genetics (rare alleles) not modeled

2. **Data Availability**
   - Forensic datasets small (typically <1000 samples)
   - Deep learning needs 1000s+ samples; feature-based approach more suitable
   - Heavy class imbalance (more 2-3 contributor cases than 8-10)

3. **Generalization**
   - Model trained on lab-controlled data; casework may have contamination, degradation
   - Kit-specific: retraining needed for new platforms (GlobalFiler, Yfiler, etc.)
   - Distribution shift on real casework

4. **Feature Limitations**
   - Hardcoded peak threshold (50 RFU); varies by equipment and kit
   - No uncertainty quantification (hard predictions only)
   - Features assume linear pixel-to-BP scaling

## Best Practices

### Data Preparation
1. **Stratify**: Ensure balanced NoC distribution in train/val/test
2. **Validation Split**: Use 20% for validation (not early stopping)
3. **Test Set**: Hold out final test set; evaluate only once
4. **Labeling**: Use manual or automated labeling; double-check edge cases

### Model Training
1. **Hyperparameter Tuning**: Grid search learning rate, batch size, dropout
2. **Feature Importance**: Analyze which features drive predictions (using SHAP/permutation)
3. **Class Weights**: Use weighted loss if classes highly imbalanced
4. **Ensemble**: Train multiple models on different folds; average predictions

### Evaluation
1. **Per-Class Metrics**: Focus on recall for high-NoC classes (rare in real casework)
2. **Confusion Matrix**: Visualize error patterns (e.g., 2-contributor often confused with 3?)
3. **Off-by-One Error**: Report "acceptable" errors (±1 contributor) separately
4. **Confidence Intervals**: Report max probability as confidence score

## Advanced Usage

### Custom Feature Extraction
```python
from DNAnet.data.preprocessing.mixture_features import extract_mixture_features

hidimage = load_hid('mixture_001.hid')
features = extract_mixture_features(
    hidimage,
    panel,
    peak_threshold=50.0,
    n_top_loci=17
)
```

### Transfer Learning
```python
# Load pre-trained model trained on large dataset
model = NoCClassifier(...)
model.load('path/to/pretrained')

# Fine-tune on small lab-specific dataset
small_dataset = load_local_mixtures(...)
model.fit(
    train_features=small_dataset['train']['features'],
    train_labels=small_dataset['train']['labels'],
    learning_rate=1e-4,  # Lower LR for fine-tuning
    epochs=10,
)
```

### Synthetic Data Augmentation
```python
# Blend single-source profiles to simulate mixtures
def create_synthetic_mixture(profile1, profile2, profile3, mixing_ratios):
    """Blend 2-3 single-source profiles with random mixing ratios."""
    mixture = (
        mixing_ratios[0] * profile1 +
        mixing_ratios[1] * profile2 +
        mixing_ratios[2] * profile3
    )
    return mixture  # Label: NoC=3
```

## Troubleshooting

**Q: Low accuracy on test set despite good validation accuracy**
- A: Likely data distribution mismatch. Check test set NoC distribution; retrain with synthetic data.

**Q: Model predicts all samples as NoC=2**
- A: Class imbalance. Use weighted loss: `weighted_loss = CrossEntropyLoss(weight=[...])` based on class frequencies.

**Q: Feature extraction takes too long**
- A: Enable caching: `dataset = NoCDataset(..., use_cache=True)`. First run is slow; subsequent runs use cache.

**Q: Memory error during training**
- A: Reduce batch size (--batch-size 16), or use gradient accumulation. For very large datasets, use data streaming.

**Q: How to interpret probabilities?**
- A: Model outputs softmax over 10 classes. For prediction, take argmax. For confidence, use max probability. For uncertainty, check entropy of distribution.

## References

### Papers
- Peak Detection in Electropherograms: [relevant papers]
- DNA Mixture Interpretation: [relevant papers]

### Related Code
- [DNANet Allele Calling](DNAnet/allele_callers.py) — Alternative approach using segmentation
- [Feature Extraction](DNAnet/data/preprocessing/mixture_features.py) — Statistical features
- [Evaluation Metrics](DNAnet/evaluation/noc_metrics.py) — Classification metrics

## Contributing

To extend the NoC module:

1. **Add new features**: Modify `extract_mixture_features()` in `mixture_features.py`
2. **Novel architectures**: Create new classifier in `models/` (inherit from `TrainableModel`)
3. **Custom metrics**: Add to `evaluation/noc_metrics.py`
4. **Bug reports**: Create issues with reproducible examples

## License

Same as DNANet [LICENSE](../LICENSE)
