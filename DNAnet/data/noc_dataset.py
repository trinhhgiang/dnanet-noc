"""
NoC (Number of Contributors) dataset for mixture prediction.

Extends HIDDataset with NoC labels and feature extraction.
"""

from typing import Dict, Optional, Sequence, Any, Union
from pathlib import Path
import json
import csv
import pandas as pd
import numpy as np
import torch

from DNAnet.data.data_models import HIDDataset, HIDImage
from DNAnet.data.preprocessing.mixture_features import extract_mixture_features


class NoCDataset(HIDDataset):
    """
    Dataset for NoC prediction from mixture electropherograms.
    
    Extends HIDDataset to:
    1. Load NoC labels (ground truth contributor counts)
    2. Extract mixture features per image
    3. Support train/test split with stratification on NoC
    """
    
    def __init__(
        self,
        root_dir: Union[str, Path],
        noc_labels_path: Union[str, Path],
        strategy: Optional[Any] = None,
        use_cache: bool = False,
        feature_normalize: bool = True,
        **kwargs
    ):
        """
        Args:
            root_dir: Root directory containing .hid files
            noc_labels_path: Path to CSV with columns [filename, noc]
            strategy: DatasetStrategy for loading (if None, defaults to basic)
            use_cache: Whether to use .arrow caching
            feature_normalize: Whether to normalize features (zero mean, unit var)
        """
        super().__init__(root_dir, strategy=strategy, use_cache=use_cache, **kwargs)
        
        self.noc_labels_path = Path(noc_labels_path)
        self.feature_normalize = feature_normalize
        
        # Load NoC labels
        self.noc_labels = self._load_noc_labels()
        
        # Track feature statistics for normalization
        self.feature_stats = None
        self.feature_names = None
    
    def _load_noc_labels(self) -> Dict[str, int]:
        """Load NoC ground truth labels from CSV or JSON."""
        if not self.noc_labels_path.exists():
            raise FileNotFoundError(f"NoC labels file not found: {self.noc_labels_path}")
        
        labels = {}
        
        if self.noc_labels_path.suffix == '.csv':
            # Load CSV: expected columns [filename, noc]
            df = pd.read_csv(self.noc_labels_path)
            for _, row in df.iterrows():
                filename = Path(row['filename']).name  # Handle both full paths and basenames
                noc = int(row['noc'])
                labels[filename] = noc
        
        elif self.noc_labels_path.suffix == '.json':
            # Load JSON: expected format {filename: noc}
            with open(self.noc_labels_path) as f:
                labels = {k: int(v) for k, v in json.load(f).items()}
        
        else:
            raise ValueError(f"Unsupported labels format: {self.noc_labels_path.suffix}")
        
        return labels
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get dataset item with features and NoC label.
        
        Returns:
            {
                'features': np.array of extracted features,
                'noc': int (number of contributors),
                'metadata': dict with original_path, etc.
            }
        """
        # Get HIDImage from parent class
        hidimage = super().__getitem__(idx)
        
        # Get filename for label lookup
        filename = Path(hidimage.path).name
        if filename not in self.noc_labels:
            # Try without extension
            filename_stem = Path(hidimage.path).stem
            if filename_stem not in self.noc_labels:
                raise KeyError(f"No NoC label found for {filename} or {filename_stem}")
            filename = filename_stem
        
        noc = self.noc_labels[filename]
        
        # Extract mixture features
        features = extract_mixture_features(
            hidimage,
            panel=self.strategy.panel if hasattr(self.strategy, 'panel') else None
        )
        
        # Normalize if needed
        if self.feature_normalize and self.feature_stats is not None:
            features = self._normalize_features(features)
        
        # Convert to array for torch compatibility
        features_array = self._features_dict_to_array(features)
        
        return {
            'features': features_array,
            'noc': noc,
            'metadata': {
                'path': str(hidimage.path),
                'filename': filename,
            }
        }
    
    def compute_feature_statistics(self) -> None:
        """
        Compute mean/std for feature normalization.
        Should be called on training set before normalizing.
        """
        all_features = []
        
        for i in range(len(self)):
            try:
                item = {
                    'features': extract_feature_dict_from_item(self[i]),
                    'noc': self[i]['noc']
                }
                all_features.append(item['features'])
            except Exception as e:
                print(f"Warning: Could not compute stats for idx {i}: {e}")
                continue
        
        if not all_features:
            raise RuntimeError("No valid features computed for statistics")
        
        # Stack all feature dicts
        all_dicts = []
        for feat_dict in all_features:
            all_dicts.append(feat_dict)
        
        # Compute per-feature statistics
        first_dict = all_dicts[0]
        self.feature_names = list(first_dict.keys())
        
        feature_arrays = {key: [] for key in self.feature_names}
        for feat_dict in all_dicts:
            for key in self.feature_names:
                feature_arrays[key].append(feat_dict.get(key, 0.0))
        
        self.feature_stats = {}
        for key in self.feature_names:
            arr = np.array(feature_arrays[key])
            self.feature_stats[key] = {
                'mean': float(np.mean(arr)),
                'std': float(np.std(arr) + 1e-6)  # Add epsilon to avoid division by zero
            }
    
    def _normalize_features(self, features: Dict[str, float]) -> Dict[str, float]:
        """Normalize features using training set statistics."""
        if self.feature_stats is None:
            return features
        
        normalized = {}
        for key, value in features.items():
            if key in self.feature_stats:
                stats = self.feature_stats[key]
                normalized[key] = (value - stats['mean']) / stats['std']
            else:
                normalized[key] = value
        
        return normalized
    
    def _features_dict_to_array(self, features: Dict[str, float]) -> np.ndarray:
        """Convert feature dictionary to numpy array with fixed order."""
        if self.feature_names is None:
            self.feature_names = sorted(features.keys())
        
        return np.array([features.get(key, 0.0) for key in self.feature_names], dtype=np.float32)
    
    def get_feature_names(self) -> Sequence[str]:
        """Return ordered list of feature names."""
        if self.feature_names is None:
            # Extract from first item
            _ = self[0]
        return self.feature_names
    
    def stratified_train_test_split(
        self,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> tuple:
        """
        Split dataset into train/test with stratification on NoC.
        Ensures balanced NoC distribution in both sets.
        
        Returns:
            (train_indices, test_indices)
        """
        from sklearn.model_selection import train_test_split
        
        # Get NoC labels for all samples
        nocs = []
        files = []
        for hidimage in self.images:
            filename = Path(hidimage.path).name
            if filename in self.noc_labels:
                nocs.append(self.noc_labels[filename])
                files.append(filename)
        
        if not files:
            raise RuntimeError("No matched files with NoC labels")
        
        # Stratified split
        train_files, test_files = train_test_split(
            files,
            test_size=test_size,
            stratify=nocs,
            random_state=random_state
        )
        
        # Convert to indices
        train_indices = [
            i for i, img in enumerate(self.images)
            if Path(img.path).name in set(train_files)
        ]
        test_indices = [
            i for i, img in enumerate(self.images)
            if Path(img.path).name in set(test_files)
        ]
        
        return train_indices, test_indices


def extract_feature_dict_from_item(item: Dict[str, Any]) -> Dict[str, float]:
    """Helper: extract feature dict from dataset item (for statistics computation)."""
    # Reconstruct dict from array if needed
    # This is a placeholder; actual implementation depends on feature names
    return {}
