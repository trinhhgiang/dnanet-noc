"""
Quick-Start Notebook Cell Templates for Kaggle
=====================================================

Copy these cell contents directly into your Kaggle notebook for quick setup.

"""

# ============================================================================
# CELL 1: Setup (Run first)
# ============================================================================

"""
# Setup & Dependencies

!pip install -q torch scikit-learn pandas numpy scipy matplotlib seaborn > /dev/null 2>&1

import sys
sys.path.insert(0, '/kaggle/input/dnanet-noc-mixture/')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch

print("✅ Dependencies installed")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
"""


# ============================================================================
# CELL 2: Load Data & Explore
# ============================================================================

"""
# Load and Explore Dataset

import numpy as np
import json

# Load features and labels
features = np.load('/kaggle/input/dnanet-noc-mixture/features.npy')
labels = np.load('/kaggle/input/dnanet-noc-mixture/labels.npy')

with open('/kaggle/input/dnanet-noc-mixture/feature_names.json') as f:
    feature_names = json.load(f)

with open('/kaggle/input/dnanet-noc-mixture/noc_labels.csv') as f:
    samples_df = pd.read_csv(f)

# Explore data
print(f"📊 Dataset Shape:")
print(f"  Samples: {features.shape[0]}")
print(f"  Features: {features.shape[1]}")
print(f"\\n📋 NoC Distribution:")
print(labels_series := pd.Series(labels).value_counts().sort_index())
print(f"\\n🧬 Sample Features:")
print(f"  First 5: {feature_names[:5]}")
print(f"  Feature value range: [{features.min():.3f}, {features.max():.3f}]")

# Visualize NoC distribution
plt.figure(figsize=(8, 4))
plt.bar(range(1, 11), np.bincount(labels, minlength=11)[1:], color='steelblue')
plt.xlabel('Number of Contributors (NoC)')
plt.ylabel('Count')
plt.title('Dataset NoC Distribution')
plt.xticks(range(1, 11))
plt.grid(alpha=0.3)
plt.show()
"""


# ============================================================================
# CELL 3: Train Model (Full Pipeline)
# ============================================================================

"""
# Train NoC Classifier

# Add to path and import
import sys
sys.path.insert(0, '/kaggle/input/dnanet-noc-mixture/')

# Run the full notebook
%run /kaggle/input/dnanet-noc-mixture/kaggle_notebook.py
"""


# ============================================================================
# CELL 4: Quick Training (Minimal Version)
# ============================================================================

"""
# Quick Training (Lightweight Version)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load data
features = np.load('/kaggle/input/dnanet-noc-mixture/features.npy')
labels = np.load('/kaggle/input/dnanet-noc-mixture/labels.npy') - 1  # 0-9

# Normalize
scaler = StandardScaler()
features = scaler.fit_transform(features)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, random_state=42, stratify=labels
)

# Model
class NoCNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(78, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
    
    def forward(self, x):
        return self.fc(x)

# Train
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = NoCNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_loader = DataLoader(
    TensorDataset(torch.from_numpy(X_train).float(),
                  torch.from_numpy(y_train).long()),
    batch_size=8, shuffle=True
)
test_loader = DataLoader(
    TensorDataset(torch.from_numpy(X_test).float(),
                  torch.from_numpy(y_test).long()),
    batch_size=8
)

print("Training...")
for epoch in range(30):
    model.train()
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        model.eval()
        with torch.no_grad():
            correct = sum((torch.argmax(model(X.to(device)), 1) == y.to(device)).sum() 
                         for X, y in test_loader)
            acc = correct / len(X_test)
        print(f"Epoch {epoch+1}: Accuracy = {acc:.4f}")

print("✅ Training complete!")
"""


# ============================================================================
# CELL 5: Make Predictions
# ============================================================================

"""
# Make Predictions on Test Data

import sys
sys.path.insert(0, '/kaggle/input/dnanet-noc-mixture/')

from kaggle_inference import NoCPredictor
import numpy as np
import pandas as pd

# Initialize predictor
predictor = NoCPredictor(
    model_path='/kaggle/input/dnanet-noc-mixture/model.pt',
    feature_names_path='/kaggle/input/dnanet-noc-mixture/feature_names.json'
)

# Load test data (or use your own)
features = np.load('/kaggle/input/dnanet-noc-mixture/features.npy')
labels = np.load('/kaggle/input/dnanet-noc-mixture/labels.npy')

# Get predictions
predictions = predictor.predict(features)

# Get confidence scores
results_df = predictor.predict_with_confidence(features)
results_df['true_label'] = labels

print("📊 Prediction Results:")
print(results_df.head(10))

# Calculate accuracy
accuracy = (results_df['predicted_noc'] == results_df['true_label']).mean()
print(f"\\nAccuracy: {accuracy:.4f}")
"""


# ============================================================================
# CELL 6: Probability Distribution
# ============================================================================

"""
# Analyze Probability Distributions

import sys
sys.path.insert(0, '/kaggle/input/dnanet-noc-mixture/')
from kaggle_inference import NoCPredictor
import numpy as np
import matplotlib.pyplot as plt

predictor = NoCPredictor(
    model_path='/kaggle/input/dnanet-noc-mixture/model.pt',
    feature_names_path='/kaggle/input/dnanet-noc-mixture/feature_names.json'
)

# Get probabilities
features = np.load('/kaggle/input/dnanet-noc-mixture/features.npy')
probs = predictor.predict_proba(features)

# Visualize first 5 samples
fig, axes = plt.subplots(1, 5, figsize=(15, 3))
for i in range(5):
    axes[i].bar(range(1, 11), probs[i], color='steelblue')
    axes[i].set_title(f'Sample {i+1}')
    axes[i].set_xlabel('NoC')
    axes[i].set_ylabel('Probability')
    axes[i].set_ylim(0, 1)

plt.tight_layout()
plt.show()

# Show average probability distribution
avg_prob = probs.mean(axis=0)
plt.figure(figsize=(8, 4))
plt.bar(range(1, 11), avg_prob, color='coral', edgecolor='black')
plt.xlabel('Number of Contributors')
plt.ylabel('Average Probability')
plt.title('Average NoC Probability Distribution')
plt.xticks(range(1, 11))
plt.grid(alpha=0.3, axis='y')
plt.show()
"""


# ============================================================================
# CELL 7: Confusion Matrix & Metrics
# ============================================================================

"""
# Evaluate Model Performance

import sys
sys.path.insert(0, '/kaggle/input/dnanet-noc-mixture/')
from kaggle_inference import NoCPredictor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

predictor = NoCPredictor(
    model_path='/kaggle/input/dnanet-noc-mixture/model.pt',
    feature_names_path='/kaggle/input/dnanet-noc-mixture/feature_names.json'
)

# Get data
features = np.load('/kaggle/input/dnanet-noc-mixture/features.npy')
labels = np.load('/kaggle/input/dnanet-noc-mixture/labels.npy')

# Predict
predictions = predictor.predict(features)

# Confusion matrix
cm = confusion_matrix(labels, predictions, labels=range(1, 11))

# Visualize
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=range(1, 11), yticklabels=range(1, 11))
plt.xlabel('Predicted NoC')
plt.ylabel('True NoC')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

# Classification report
print(classification_report(labels, predictions, labels=range(1, 11)))
"""


# ============================================================================
# CELL 8: Analyze Misclassifications
# ============================================================================

"""
# Identify & Analyze Misclassifications

import sys
sys.path.insert(0, '/kaggle/input/dnanet-noc-mixture/')
from kaggle_inference import NoCPredictor
import numpy as np
import pandas as pd

predictor = NoCPredictor(
    model_path='/kaggle/input/dnanet-noc-mixture/model.pt',
    feature_names_path='/kaggle/input/dnanet-noc-mixture/feature_names.json'
)

# Get data
features = np.load('/kaggle/input/dnanet-noc-mixture/features.npy')
labels = np.load('/kaggle/input/dnanet-noc-mixture/labels.npy')

# Predict with confidence
results = predictor.predict_with_confidence(features)
results['true_label'] = labels

# Find misclassifications
misclass = results[results['predicted_noc'] != results['true_label']]

print(f"📊 Misclassification Summary:")
print(f"Total misclassified: {len(misclass)}/{len(results)}")
print(f"Error rate: {100 * len(misclass) / len(results):.1f}%")
print(f"\\n🔍 Misclassified samples:")
print(misclass[['sample_id', 'true_label', 'predicted_noc', 'confidence']])

# Average confidence for correct vs incorrect
correct = results[results['predicted_noc'] == results['true_label']]
print(f"\\nAverage confidence (correct): {correct['confidence'].mean():.4f}")
print(f"Average confidence (incorrect): {misclass['confidence'].mean():.4f}")
"""


# ============================================================================
# CELL 9: Feature Importance Approximation
# ============================================================================

"""
# Approximate Feature Importance

import sys
sys.path.insert(0, '/kaggle/input/dnanet-noc-mixture/')
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt

# Load features and names
features = np.load('/kaggle/input/dnanet-noc-mixture/features.npy')
with open('/kaggle/input/dnanet-noc-mixture/feature_names.json') as f:
    feature_names = json.load(f)

# Simple approximation: feature variance as proxy for importance
feature_variance = np.var(features, axis=0)
importance_df = pd.DataFrame({
    'feature': feature_names,
    'variance': feature_variance
}).sort_values('variance', ascending=False)

# Plot top features
plt.figure(figsize=(10, 6))
top_n = 15
plt.barh(range(top_n), importance_df['variance'].head(top_n).values)
plt.yticks(range(top_n), importance_df['feature'].head(top_n).values)
plt.xlabel('Feature Variance')
plt.title('Top 15 Important Features (by variance)')
plt.tight_layout()
plt.show()

print("Top 10 features:")
print(importance_df.head(10))
"""


# ============================================================================
# CELL 10: Export Results
# ============================================================================

"""
# Export Results to CSV

import sys
sys.path.insert(0, '/kaggle/input/dnanet-noc-mixture/')
from kaggle_inference import NoCPredictor
import numpy as np
import pandas as pd

predictor = NoCPredictor(
    model_path='/kaggle/input/dnanet-noc-mixture/model.pt',
    feature_names_path='/kaggle/input/dnanet-noc-mixture/feature_names.json'
)

features = np.load('/kaggle/input/dnanet-noc-mixture/features.npy')
labels = np.load('/kaggle/input/dnanet-noc-mixture/labels.npy')

# Get predictions with confidence
results = predictor.predict_with_confidence(features)
results['true_label'] = labels

# Save to CSV
results.to_csv('/kaggle/output/predictions.csv', index=False)

print("✅ Saved predictions to /kaggle/output/predictions.csv")
print(results)
"""


if __name__ == '__main__':
    print("""
    📋 Kaggle Quick-Start Templates
    ================================
    
    Copy cell contents into your Kaggle notebook:
    
    1. CELL 1: Setup & Dependencies
    2. CELL 2: Load & Explore Data  
    3. CELL 3: Train Full Model
    4. CELL 4: Quick Training (Lightweight)
    5. CELL 5: Make Predictions
    6. CELL 6: Probability Distributions
    7. CELL 7: Confusion Matrix & Metrics
    8. CELL 8: Analyze Misclassifications
    9. CELL 9: Feature Importance
    10. CELL 10: Export Results
    
    Start with Cell 1, then choose your workflow!
    """)
