"""
Utility script to create synthetic DNA mixtures for NoC training/testing.

This script blends single-source DNA profiles to create synthetic mixtures
with known contributor counts, useful for training the NoC classifier
on limited real data.
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd
from PIL import Image


def load_hid_electropherogram(hid_path: Path) -> np.ndarray:
    """
    Load electropherogram data from HID file.
    
    Returns: np.ndarray of shape (n_dyes, n_scanpoints)
    """
    # Placeholder: actual loading uses parse_raw_hid.py
    # For now, assume data is pre-extracted to numpy format
    if hid_path.suffix == '.npy':
        return np.load(hid_path)
    else:
        raise NotImplementedError("Implement HID parsing via ABIF library")


def create_synthetic_mixture(
    epgs: List[np.ndarray],
    mixing_ratios: np.ndarray,
    noise_level: float = 0.05,
) -> np.ndarray:
    """
    Create synthetic mixture by blending single-source electropherograms.
    
    Args:
        epgs: List of electropherogram arrays (each shape: n_dyes x n_scanpoints)
        mixing_ratios: Array of mixing ratios (must sum to 1.0)
        noise_level: Gaussian noise level (fraction of max signal)
    
    Returns:
        Synthetic mixture electropherogram
    """
    if len(epgs) != len(mixing_ratios):
        raise ValueError(f"Mismatch: {len(epgs)} EPGs but {len(mixing_ratios)} ratios")
    
    if not np.isclose(np.sum(mixing_ratios), 1.0):
        raise ValueError(f"Mixing ratios must sum to 1.0, got {np.sum(mixing_ratios)}")
    
    # Ensure all EPGs have same shape
    shape = epgs[0].shape
    for epg in epgs:
        if epg.shape != shape:
            raise ValueError(f"All EPGs must have same shape; got {shape} and {epg.shape}")
    
    # Blend EPGs with mixing ratios
    mixture = np.zeros_like(epgs[0], dtype=float)
    for epg, ratio in zip(epgs, mixing_ratios):
        mixture += ratio * epg.astype(float)
    
    # Add Gaussian noise
    if noise_level > 0:
        noise = np.random.normal(
            0,
            noise_level * np.max(mixture),
            mixture.shape
        )
        mixture = np.maximum(mixture + noise, 0)  # Ensure non-negative
    
    return mixture.astype(np.uint8)


def generate_synthetic_dataset(
    input_dir: Path,
    output_dir: Path,
    noc_values: Tuple[int] = (2, 3, 4, 5),
    n_mixtures_per_noc: int = 20,
    seed: int = 42,
):
    """
    Generate synthetic mixture dataset from single-source profiles.
    
    Args:
        input_dir: Directory with single-source .hid or .npy files
        output_dir: Output directory for synthetic mixtures
        noc_values: Tuple of NoC values to generate (e.g., (2, 3, 4, 5))
        n_mixtures_per_noc: Number of mixtures per NoC value
        seed: Random seed for reproducibility
    """
    np.random.seed(seed)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading single-source profiles from {input_dir}")
    
    # Load all single-source profiles
    epg_files = list(input_dir.glob('*.hid')) + list(input_dir.glob('*.npy'))
    if not epg_files:
        raise FileNotFoundError(f"No .hid or .npy files in {input_dir}")
    
    profiles = []
    for fpath in epg_files[:50]:  # Limit to 50 profiles
        try:
            epg = load_hid_electropherogram(fpath)
            profiles.append(epg)
        except Exception as e:
            print(f"  Warning: Could not load {fpath}: {e}")
            continue
    
    print(f"Loaded {len(profiles)} single-source profiles")
    
    # Generate synthetic mixtures
    noc_labels = []
    synthetic_count = 0
    
    for noc in noc_values:
        print(f"\nGenerating mixtures with NoC={noc}")
        
        for mixture_idx in range(n_mixtures_per_noc):
            # Randomly select 'noc' profiles
            selected_indices = np.random.choice(len(profiles), noc, replace=False)
            epgs = [profiles[i] for i in selected_indices]
            
            # Random mixing ratios (Dirichlet distribution)
            mixing_ratios = np.random.dirichlet(np.ones(noc))
            
            # Create mixture
            try:
                mixture = create_synthetic_mixture(epgs, mixing_ratios, noise_level=0.02)
            except Exception as e:
                print(f"  Warning: Could not create mixture {mixture_idx}: {e}")
                continue
            
            # Save mixture
            output_filename = f"synthetic_noc{noc}_mix{mixture_idx:03d}.npy"
            output_path = output_dir / output_filename
            np.save(output_path, mixture)
            
            noc_labels.append({
                'filename': output_filename,
                'noc': noc,
                'donors': selected_indices.tolist(),
                'mixing_ratios': mixing_ratios.tolist(),
            })
            
            synthetic_count += 1
            if (mixture_idx + 1) % max(1, n_mixtures_per_noc // 5) == 0:
                print(f"  Generated {mixture_idx + 1}/{n_mixtures_per_noc}")
    
    # Save labels
    labels_df = pd.DataFrame([
        {
            'filename': item['filename'],
            'noc': item['noc'],
        }
        for item in noc_labels
    ])
    labels_path = output_dir / 'noc_labels.csv'
    labels_df.to_csv(labels_path, index=False)
    
    # Save detailed metadata
    metadata_path = output_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(noc_labels, f, indent=2)
    
    print(f"\n✓ Generated {synthetic_count} synthetic mixtures")
    print(f"✓ Labels saved to {labels_path}")
    print(f"✓ Metadata saved to {metadata_path}")
    
    return labels_df


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Generate synthetic DNA mixtures for NoC training'
    )
    parser.add_argument(
        '--input-dir',
        required=True,
        help='Directory with single-source .hid or .npy files'
    )
    parser.add_argument(
        '--output-dir',
        required=True,
        help='Output directory for synthetic mixtures'
    )
    parser.add_argument(
        '--noc-values',
        type=int,
        nargs='+',
        default=[2, 3, 4, 5],
        help='NoC values to generate (default: 2 3 4 5)'
    )
    parser.add_argument(
        '--n-per-noc',
        type=int,
        default=20,
        help='Number of mixtures per NoC value (default: 20)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    
    generate_synthetic_dataset(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        noc_values=tuple(args.noc_values),
        n_mixtures_per_noc=args.n_per_noc,
        seed=args.seed,
    )
