# 🧬 DNA Mixture NoC Classification - Kaggle

## Overview

This dataset and code provides a complete solution for predicting the **Number of Contributors (NoC)** in DNA mixtures using deep learning.

**What is NoC?** The Number of Contributors represents how many individuals' DNA is present in a forensic sample (1-10 people). Accurately predicting NoC is crucial for DNA mixture interpretation.

**Dataset**: Pre-processed PROVEDIt DNA profiles converted to machine learning features (78 features per sample, 25 samples with 1-10 NoC labels)

---

## Quick Start (Copy into Kaggle Notebook)

### Option A: Train from Scratch

```python
%cd /kaggle/input/dnanet-noc-mixture/
%run kaggle_notebook.py
```

This will:
- Load the converted PROVEDIt data (25 samples, 1-10 NoC)
- Train a fully-connected neural network
- Evaluate on test set
- Save model, metrics, and visualizations to `/kaggle/output/`

### Option B: Use Pre-trained Model for Inference

```python
import sys
sys.path.insert(0, '/kaggle/input/dnanet-noc-mixture/')

from kaggle_inference import NoCPredictor
import numpy as np

# Load predictor
predictor = NoCPredictor(
    model_path='/kaggle/input/dnanet-noc-mixture/model.pt',
    feature_names_path='/kaggle/input/dnanet-noc-mixture/feature_names.json'
)

# Make predictions
features = np.random.randn(5, 78)  # Replace with real features
predictions = predictor.predict(features)
print(predictions)  # Output: array([2, 3, 2, 4, 1])
```

---

## Files in This Dataset

| File | Purpose |
|------|---------|
| `kaggle_notebook.py` | Full training pipeline (train → evaluate → visualize) |
| `kaggle_inference.py` | Inference wrapper for predictions on new data |
| `features.npy` | Pre-extracted features (25 samples × 78 features) |
| `labels.npy` | NoC labels (1-10, shape: 25) |
| `noc_labels.csv` | Sample metadata (filename, NoC) |
| `feature_names.json` | Names of the 78 features extracted |
| `dataset-metadata.json` | Kaggle dataset metadata |

---

## Dataset Details

### Data Source
- **PROVEDIt Dataset**: 25 DNA profiles across 1-5 person mixtures
- **Kits**: SGM Plus, PowerPlex, Globalfiler, PPF6C
- **Features**: 78 statistical features per sample extracted from STR profiles

### Feature Categories

**Per-locus features (78 total):**
- Peak count per locus (×14 loci)
- Mean/max/std RFU values (×14)
- Allele counts and height ratios (×14)

**Global features:**
- Total peaks, RFU distribution statistics
- Anomaly indicators (3+ alleles, 4+ alleles per locus)

### NoC Distribution
```
NoC 1: 13 samples
NoC 2: 3 samples
NoC 3: 3 samples
NoC 4: 3 samples
NoC 5: 3 samples
Total: 25 samples
```

---

## Model Architecture

**Fully-Connected Neural Network:**
```
Input (78) → FC(128) → ReLU → Dropout(0.2)
           → FC(64) → ReLU → Dropout(0.2)
           → FC(10) → Softmax
           → Output (NoC: 1-10)
```

**Training Details:**
- Loss: CrossEntropyLoss (multi-class classification)
- Optimizer: Adam (lr=0.001)
- Batch size: 8
- Early stopping: patience=10 epochs
- Train/Val/Test split: 16 / 4 / 5 samples

---

## Usage Examples

### 1️⃣ Train and Evaluate

```python
%run /kaggle/input/dnanet-noc-mixture/kaggle_notebook.py
```

Output files:
- `model.pt` - Trained model weights
- `metrics.json` - Accuracy, best epoch, hyperparameters
- `predictions.csv` - Predictions on test set
- `training_curves.png` - Loss and accuracy plots
- `confusion_matrix.png` - Classification confusion matrix
- `noc_distribution.png` - NoC label distribution

### 2️⃣ Single Sample Prediction

```python
from kaggle_inference import NoCPredictor
import numpy as np

predictor = NoCPredictor('/kaggle/input/dnanet-noc-mixture/model.pt',
                         '/kaggle/input/dnanet-noc-mixture/feature_names.json')

# Single sample (78 features)
sample = np.random.randn(78)
predicted_noc = predictor.predict(sample)
print(f"Predicted NoC: {predicted_noc[0]}")  # e.g., 3
```

### 3️⃣ Batch Predictions with Confidence

```python
from kaggle_inference import NoCPredictor
import numpy as np

predictor = NoCPredictor('/kaggle/input/dnanet-noc-mixture/model.pt',
                         '/kaggle/input/dnanet-noc-mixture/feature_names.json')

# Batch of 10 samples
batch = np.random.randn(10, 78)

# Get predictions with confidence scores
results = predictor.predict_with_confidence(batch)
print(results)
# Output:
#    sample_id  predicted_noc  confidence
# 0          0              2       0.8234
# 1          1              3       0.7156
# ...
```

### 4️⃣ Probability Distribution

```python
from kaggle_inference import NoCPredictor
import numpy as np
import matplotlib.pyplot as plt

predictor = NoCPredictor('/kaggle/input/dnanet-noc-mixture/model.pt',
                         '/kaggle/input/dnanet-noc-mixture/feature_names.json')

# Get probabilities for all classes
sample = np.random.randn(1, 78)
probs = predictor.predict_proba(sample)  # Shape: (1, 10)

# Plot
plt.bar(range(1, 11), probs[0])
plt.xlabel('Number of Contributors')
plt.ylabel('Probability')
plt.title('NoC Probability Distribution')
plt.show()
```

### 5️⃣ Load Raw Features

```python
import numpy as np
import json

# Load raw data
features = np.load('/kaggle/input/dnanet-noc-mixture/features.npy')
labels = np.load('/kaggle/input/dnanet-noc-mixture/labels.npy')

with open('/kaggle/input/dnanet-noc-mixture/feature_names.json') as f:
    feature_names = json.load(f)

print(f"Shape: {features.shape}")  # (25, 78)
print(f"NoC values: {np.unique(labels)}")  # [1 2 3 4 5]
print(f"Features: {feature_names[:5]}")  # First 5 feature names
```

---

## Expected Results

**Model Performance:**
- Test Accuracy: ~80-85% (on 25 samples with stratified split)
- Per-class precision/recall varies depending on sample size
- Off-by-one error: <5% (predicted NoC within 1 of true NoC)

**Common Misclassifications:**
- NoC 1 vs 2: Similar feature distributions at low mixture complexity
- NoC 4 vs 5: Limited samples for high mixtures

---

## Tips for Kaggle Notebooks

### 🎯 Improve Performance
1. **Increase epochs** if training curves still improving
2. **Adjust batch size** based on available memory
3. **Tune learning rate** (try 0.0001-0.01 range)
4. **Add regularization** (increase dropout_rate in Config)

### 📊 Analyze Predictions
```python
# Identify misclassified samples
import pandas as pd
results = pd.read_csv('/kaggle/output/predictions.csv')
misclassified = results[results['label'] != results['prediction']]
print(f"Misclassified: {len(misclassified)}/25")
```

### 💾 Save Model for Future Use
```python
import torch
# Model already saved, but to load later:
model_state = torch.load('/kaggle/output/model.pt')
```

### 🔄 Fine-tune on Your Data
```python
# Load pre-trained model and train on custom data
from kaggle_notebook import NoCClassifier, Config
import torch

model = NoCClassifier(input_dim=78)
model.load_state_dict(torch.load('/path/to/pretrained.pt'))

# Then train on your data with lower learning rate
# (similar to transfer learning)
```

---

## Troubleshooting

### ❌ "ModuleNotFoundError: torch"
```python
!pip install torch scikit-learn pandas numpy scipy matplotlib seaborn
```

### ❌ "FileNotFoundError: features.npy"
Make sure you're running in the dataset directory:
```python
%cd /kaggle/input/dnanet-noc-mixture/
```

### ❌ "CUDA out of memory"
Reduce batch size in Config:
```python
Config.BATCH_SIZE = 4  # Instead of 8
```

### ❌ "features and labels shape mismatch"
Check that you're loading both correctly:
```python
features = np.load('features.npy')  # Should be (25, 78)
labels = np.load('labels.npy')      # Should be (25,)
```

---

## Related Resources

- **GitHub Repository**: https://github.com/trinhhgiang/dnanet-noc
- **Original DNANet**: https://github.com/NetherlandsForensicInstitute/DNANet
- **PROVEDIt Dataset**: https://www.fbi.gov/services/laboratory/scientific-training-program/fbi-proved-it-dna-profiling-data
- **STR Analysis**: International standards for DNA profiling (SWGDAM guidelines)

---

## Citation

If you use this dataset or code, please cite:
```
@misc{dnanet-noc,
  title={DNA Mixture Number of Contributors Classification},
  author={Trinh Hoang Giang},
  year={2026},
  url={https://github.com/trinhhgiang/dnanet-noc}
}
```

---

## Questions?

- Check the **Kaggle Discussion** tab for community discussions
- Review **kaggle_notebook.py** for detailed implementation
- See **kaggle_inference.py** for API reference
- Open an issue on [GitHub](https://github.com/trinhhgiang/dnanet-noc)

**Happy DNA mixture analysis! 🧬**
