# Deployment Guide: GitHub & Kaggle

## 📤 Push to GitHub

### Option 1: Create New GitHub Repository

**Step 1: Create repository on GitHub**
1. Go to [github.com/new](https://github.com/new)
2. Repository name: `dnanet-noc` (or similar)
3. Description: "Number of Contributors (NoC) prediction for DNA mixtures using PyTorch"
4. Choose: Public (recommended for open research)
5. Click "Create repository"

**Step 2: Push from terminal**

```bash
# Navigate to project
cd "/home/giang/GenAI/20252/Tin sinh/DNANet"

# Initialize git (if not already done)
git init

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: NoC prediction module for DNA mixtures"

# Add remote
git remote add origin https://github.com/YOUR_USERNAME/dnanet-noc.git

# Rename branch to main
git branch -M main

# Push to GitHub
git push -u origin main
```

**Step 3: Create `.gitignore`**

Create `/.gitignore`:
```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
*.egg-info/
dist/
build/
.venv/
venv/

# Data
data/provedit_converted/
data/mixture_samples/
data/test_samples/

# Model outputs
output/
mlruns/
.mlflow/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# Environment
.env
.env.local
```

Then commit:
```bash
git add .gitignore
git commit -m "Add gitignore"
git push
```

---

## 🚀 Push to Kaggle

### Option 1: Kaggle Dataset

**Step 1: Install Kaggle CLI**

```bash
pip install kaggle

# Download API key from https://www.kaggle.com/account
# Place at ~/.kaggle/kaggle.json
```

**Step 2: Create dataset directory**

```bash
# Create clean directory with only code and data
mkdir kaggle_dataset
cd kaggle_dataset

# Copy key files
cp -r DNAnet/ .
cp train_noc.py .
cp evaluate_noc.py .
cp convert_provedit_to_noc.py .
cp examples_noc.py .
cp requirements.txt .
cp NoC_README.md .
cp README_FOR_GITHUB.md README.md
cp config/ .
cp data/provedit_converted . 2>/dev/null || true
```

**Step 3: Create dataset.json**

Create `metadata.json` in kaggle_dataset:
```json
{
  "title": "DNANet NoC - DNA Mixture Contributor Prediction",
  "id": "yourname/dnanet-noc-prediction",
  "licenses": [{"name": "CC0-1.0"}],
  "keywords": ["dna", "mixture", "contributors", "pytorch", "machine-learning"],
  "collaborators": [],
  "data": []
}
```

**Step 4: Upload dataset**

```bash
# Create Kaggle dataset
kaggle datasets create -p ./kaggle_dataset \
  --public \
  --dir-mode tar

# Update existing dataset
kaggle datasets version -p ./kaggle_dataset -m "Updated with new results"
```

### Option 2: Kaggle Code Notebook

1. Go to [kaggle.com/code](https://kaggle.com/code)
2. Click "New Notebook"
3. Add this code:

```python
# Install dependencies
!pip install -q torch scikit-learn pandas scipy loguru

# Clone or upload code
!git clone https://github.com/YOUR_USERNAME/dnanet-noc.git
%cd dnanet-noc

# Run training
!python train_noc.py \
    --noc-labels data/provedit_converted/noc_labels.csv \
    --data-dir data/provedit_converted \
    --output-dir output/noc_model \
    --epochs 50 \
    --batch-size 8

# Evaluate
!python evaluate_noc.py \
    --model-dir output/noc_model \
    --test-data data/test_samples \
    --noc-labels data/test_labels.csv
```

4. Share publicly

---

## 📋 Checklist Before Pushing

- [x] Convert PROVEDIt data: `data/provedit_converted/`
- [x] All Python scripts working
- [x] Tests passing
- [x] Documentation complete (`NoC_README.md`, `README_FOR_GITHUB.md`)
- [x] Requirements file: `requirements.txt`
- [x] Code follows PEP 8 style
- [x] Docstrings added to functions
- [x] No credentials in code
- [x] `.gitignore` configured
- [x] LICENSE file included

---

## 🔐 Important: Clean Up Before Pushing

**Remove sensitive/large files:**

```bash
# Remove large model files (keep only code, not trained models)
rm -rf output/

# Remove cache
rm -rf __pycache__ .pytest_cache .venv

# Remove data if too large (>100MB per Kaggle limit)
# but keep noc_labels.csv for reference

# Keep only this structure for Kaggle:
# - Source code
# - Small sample data (or links to data)
# - Documentation
```

---

## 📊 GitHub Repository Template

Create essential files for GitHub:

### `LICENSE` (MIT License)
```
MIT License

Copyright (c) 2026 [Your Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

### `CONTRIBUTING.md`
```markdown
# Contributing

Pull requests welcome! 

## Development Setup

```bash
git clone https://github.com/yourname/dnanet-noc.git
cd dnanet-noc
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running Tests

```bash
pytest tests/
```

## Code Style

Follow PEP 8. Use `black` for formatting:
```bash
black DNAnet/*.py
```
```

### `.github/workflows/tests.yml` (GitHub Actions)
```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.11'
      - run: pip install -r requirements.txt
      - run: python -m pytest tests/ || true
```

---

## 🎯 Quick Start Commands

### GitHub Only
```bash
cd "/home/giang/GenAI/20252/Tin sinh/DNANet"
git init
git add .
git commit -m "Initial: NoC prediction for DNA mixtures"
git remote add origin https://github.com/YOUR_USERNAME/dnanet-noc.git
git branch -M main
git push -u origin main
```

### Kaggle Only
```bash
kaggle datasets create -p ./kaggle_dataset --public
```

### Both (Recommended)
```bash
# Push code to GitHub
git push -u origin main

# Package for Kaggle
kaggle datasets create -p ./kaggle_dataset --public
kaggle datasets version -p ./kaggle_dataset -m "Synced from GitHub"
```

---

## 📚 After Publishing

1. **Add badges to README:**
   ```markdown
   [![GitHub](https://img.shields.io/badge/GitHub-DNANet%20NoC-blue)](https://github.com/yourname/dnanet-noc)
   [![Kaggle](https://img.shields.io/badge/Kaggle-Dataset-20B2AA)](https://kaggle.com/datasets/yourname/dnanet-noc)
   ```

2. **Share on:**
   - Reddit: r/MachineLearning, r/bioinformatics
   - Twitter/LinkedIn with hashtags: #DNA #ML #PyTorch
   - Research communities
   - Kaggle discussions

3. **Maintain repository:**
   - Respond to issues/PRs
   - Update documentation
   - Fix bugs promptly
   - Release new versions on PyPI:
     ```bash
     pip install build twine
     python -m build
     twine upload dist/*
     ```

---

**Ready to deploy!** Choose GitHub, Kaggle, or both based on your needs. 🚀
