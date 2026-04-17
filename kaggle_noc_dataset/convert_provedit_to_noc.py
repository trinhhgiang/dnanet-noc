"""
PROVEDIt CSV to NoC Training Data Converter

Converts PROVEDIt STR allele calling CSVs to NoC-ready format.
Extracts features from allele patterns and generates training labels.

Usage:
    python convert_provedit_to_noc.py \
        --input-dir /path/to/PROVEDIt_1-5-Person_CSVs \
        --output-dir data/provedit_converted
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import pandas as pd
from collections import defaultdict


def extract_noc_from_path(file_path: Path) -> Optional[int]:
    """Extract NoC from directory structure."""
    parts = file_path.parts
    
    for part in parts:
        if part == '1-Person':
            return 1
        elif part == '2-Person':
            return 2
        elif part == '3-Person':
            return 3
        elif part == '4-Person':
            return 4
        elif part == '5-Person':
            return 5
    
    return None


def parse_provedit_csv(csv_path: Path) -> Dict:
    """Parse PROVEDIt CSV file and extract alleles and heights."""
    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except Exception as e:
        raise ValueError(f"Could not parse {csv_path}: {e}")
    
    alleles_per_locus = defaultdict(list)
    heights_per_locus = defaultdict(list)
    
    for _, row in df.iterrows():
        marker = row['Marker']
        
        for i in range(1, 101):
            allele_col = f'Allele {i}'
            height_col = f'Height {i}'
            
            if allele_col not in df.columns:
                break
            
            allele = row[allele_col]
            height = row[height_col]
            
            if pd.isna(allele) or allele == '' or allele == 'OL':
                continue
            
            try:
                allele_val = float(allele)
                height_val = float(height) if pd.notna(height) else 0
                alleles_per_locus[marker].append(allele_val)
                heights_per_locus[marker].append(height_val)
            except (ValueError, TypeError):
                continue
    
    total_peaks = sum(len(h) for h in heights_per_locus.values())
    
    return {
        'alleles_per_locus': dict(alleles_per_locus),
        'heights_per_locus': dict(heights_per_locus),
        'n_peaks': total_peaks,
    }


def extract_features_from_provedit(parsed_data: Dict) -> Dict[str, float]:
    """Extract mixture features from parsed PROVEDIt data."""
    features = {}
    
    heights_per_locus = parsed_data['heights_per_locus']
    
    autosomal_loci = [
        'D8S1179', 'D21S11', 'D7S820', 'CSF1PO', 'D3S1358',
        'TH01', 'D13S317', 'D16S539', 'D2S1338', 'D19S433',
        'vWA', 'D5S818', 'D18S51', 'FGA'
    ]
    
    all_heights = []
    peak_counts = []
    
    for locus in autosomal_loci:
        if locus not in heights_per_locus:
            features[f'{locus}_num_peaks'] = 0.0
            features[f'{locus}_mean_rfu'] = 0.0
            features[f'{locus}_max_rfu'] = 0.0
            features[f'{locus}_std_rfu'] = 0.0
            features[f'{locus}_height_ratio_12'] = 0.0
            peak_counts.append(0.0)
        else:
            heights = np.array(heights_per_locus[locus])
            features[f'{locus}_num_peaks'] = float(len(heights))
            features[f'{locus}_mean_rfu'] = float(np.mean(heights))
            features[f'{locus}_max_rfu'] = float(np.max(heights))
            features[f'{locus}_std_rfu'] = float(np.std(heights))
            
            if len(heights) >= 2:
                sorted_h = sorted(heights, reverse=True)
                features[f'{locus}_height_ratio_12'] = float(sorted_h[0] / (sorted_h[1] + 1e-6))
            else:
                features[f'{locus}_height_ratio_12'] = 0.0
            
            all_heights.extend(heights.tolist())
            peak_counts.append(float(len(heights)))
    
    if all_heights:
        all_heights = np.array(all_heights)
        features['total_peaks'] = float(len(all_heights))
        features['global_mean_rfu'] = float(np.mean(all_heights))
        features['global_std_rfu'] = float(np.std(all_heights))
        features['global_max_rfu'] = float(np.max(all_heights))
    else:
        features['total_peaks'] = 0.0
        features['global_mean_rfu'] = 0.0
        features['global_std_rfu'] = 0.0
        features['global_max_rfu'] = 0.0
    
    # Locus statistics
    peak_counts = np.array(peak_counts)
    features['n_loci_with_3_peaks'] = float(np.sum(peak_counts >= 3))
    features['n_loci_with_4_peaks'] = float(np.sum(peak_counts >= 4))
    features['max_peaks_any_locus'] = float(np.max(peak_counts)) if len(peak_counts) > 0 else 0.0
    features['mean_peaks_per_locus'] = float(np.mean(peak_counts))
    
    return features


def convert_provedit_dataset(input_dir: Path, output_dir: Path):
    """Convert PROVEDIt CSV files to training format."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Scanning {input_dir} for CSV files...")
    csv_files = list(input_dir.rglob('*.csv'))
    print(f"Found {len(csv_files)} CSV files")
    
    if len(csv_files) == 0:
        print("ERROR: No CSV files found!")
        return
    
    all_features = []
    all_labels = []
    metadata_list = []
    feature_names = None
    
    for i, csv_path in enumerate(csv_files):
        if (i + 1) % max(1, len(csv_files) // 10) == 0:
            print(f"  Processed {i+1}/{len(csv_files)}")
        
        try:
            noc = extract_noc_from_path(csv_path)
            if noc is None:
                continue
            
            parsed = parse_provedit_csv(csv_path)
            features = extract_features_from_provedit(parsed)
            
            if feature_names is None:
                feature_names = sorted(features.keys())
            
            features_array = [features.get(name, 0.0) for name in feature_names]
            all_features.append(features_array)
            all_labels.append(noc)
            
            metadata_list.append({
                'filename': csv_path.name,
                'noc': noc,
            })
        except Exception as e:
            continue
    
    if len(all_features) == 0:
        print("ERROR: No valid samples processed!")
        return
    
    all_features = np.array(all_features, dtype=np.float32)
    all_labels = np.array(all_labels)
    
    print(f"\n✓ Converted {len(all_labels)} samples")
    try:
        print(f"  NoC distribution: {dict(zip(*np.unique(all_labels, return_counts=True)))}")
    except:
        pass
    print(f"  Features: {len(feature_names)}")
    
    # Save
    np.save(output_dir / 'features.npy', all_features)
    np.save(output_dir / 'labels.npy', all_labels)
    
    labels_df = pd.DataFrame({
        'filename': [m['filename'] for m in metadata_list],
        'noc': all_labels
    })
    labels_df.to_csv(output_dir / 'noc_labels.csv', index=False)
    
    with open(output_dir / 'feature_names.json', 'w') as f:
        json.dump(feature_names, f, indent=2)
    
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata_list, f, indent=2)
    
    print(f"\n✓ Saved to {output_dir}:")
    print(f"  - noc_labels.csv")
    print(f"  - features.npy ({all_features.shape})")
    print(f"  - feature_names.json")
    print(f"  - metadata.json")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert PROVEDIt CSV data for NoC training'
    )
    parser.add_argument(
        '--input-dir',
        required=True,
        help='Root directory with PROVEDIt CSVs'
    )
    parser.add_argument(
        '--output-dir',
        required=True,
        help='Output directory for converted data'
    )
    args = parser.parse_args()
    
    convert_provedit_dataset(Path(args.input_dir), Path(args.output_dir))
