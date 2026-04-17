#!/usr/bin/env python3
"""
Convert ALL PROVEDIt DNA CSVs to NoC training format (expanded version)

Recursively scans ALL kits, time points, and variants to maximize sample count.
Extracts NoC from folder hierarchy and converts allele data to feature vectors.

Usage:
    python convert_provedit_all.py --input-dir <path> --output-dir <path>

Output:
    - features.npy (N × 78): Feature matrix
    - labels.npy (N,): NoC labels (1-10)
    - noc_labels.csv: Metadata
    - feature_names.json: Feature names
    - metadata.json: Conversion stats
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from scipy import stats


def extract_noc_from_path(filepath: str) -> Optional[int]:
    """
    Extract NoC label from directory path.
    
    Examples:
        /path/1-Person/file.csv → 1
        /path/2-Person/file.csv → 2
        /path/1P/file.csv → 1
        /path/2-5P/file.csv → 2-5 (returns first value or range center)
    """
    path = str(filepath).lower()
    
    # Check for X-Person folders
    for i in range(1, 11):
        if f'{i}-person' in path or f'{i}p' in path:
            return i
    
    # Check for range patterns like "2-5-Persons"
    if '2-5-person' in path or '2-5p' in path:
        return 2  # Use first value in range
    
    return None


def parse_provedit_csv(filepath: str) -> Optional[Dict]:
    """
    Parse PROVEDIt CSV file with STR allele calls.
    
    Returns dict with locus → allele list mappings
    """
    loci_data = {}
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            if reader is None or reader.fieldnames is None:
                return None
            
            for row in reader:
                if not row or len(row) < 2:
                    continue
                
                # First column is typically Locus, second is Size/Allele, third is RFU
                try:
                    locus = row.get(list(row.keys())[0], '').strip()
                    size = row.get(list(row.keys())[1], '').strip()
                    rfu_str = row.get(list(row.keys())[2], '0').strip()
                    
                    # Skip headers and invalid rows
                    if not locus or locus.lower() in ['locus', 'size', 'rfu']:
                        continue
                    if not size or size.lower() in ['size', 'rfu']:
                        continue
                    
                    try:
                        rfu = float(rfu_str)
                    except (ValueError, TypeError):
                        rfu = 0.0
                    
                    if locus not in loci_data:
                        loci_data[locus] = []
                    
                    loci_data[locus].append({
                        'allele': size,
                        'rfu': rfu
                    })
                
                except (IndexError, KeyError):
                    continue
        
        return loci_data if loci_data else None
    
    except Exception as e:
        print(f"  ⚠️  Failed to parse {filepath}: {e}")
        return None


def extract_features_from_provedit(alleles_dict: Dict) -> Optional[np.ndarray]:
    """
    Extract 78 statistical features from allele data.
    
    Returns: feature vector (78,) or None if insufficient data
    """
    if not alleles_dict or len(alleles_dict) == 0:
        return None
    
    features = []
    feature_names = []
    
    # Select up to 14 loci (prioritize common STR loci)
    priority_loci = ['AMEL', 'D3S1358', 'vWA', 'FGA', 'D5S818', 'D7S820', 'D8S1179',
                     'D13S317', 'D16S539', 'D18S51', 'D19S433', 'D21S11', 'D2S1338', 'D12S391']
    
    selected_loci = []
    for locus in priority_loci:
        if locus in alleles_dict:
            selected_loci.append(locus)
        if len(selected_loci) >= 14:
            break
    
    # Add any remaining loci
    for locus in sorted(alleles_dict.keys()):
        if locus not in selected_loci:
            selected_loci.append(locus)
        if len(selected_loci) >= 17:  # Maximum 17 loci for safety
            break
    
    # Extract per-locus features
    rfu_all = []
    allele_counts = []
    
    for locus in selected_loci:
        alleles = alleles_dict[locus]
        if not alleles:
            continue
        
        rfus = [a['rfu'] for a in alleles if a['rfu'] > 0]
        n_alleles = len(alleles)
        n_peaks = len(rfus)
        
        # Per-locus features
        features.append(n_peaks)
        feature_names.append(f'{locus}_peak_count')
        
        if rfus:
            features.append(np.mean(rfus))
            feature_names.append(f'{locus}_mean_rfu')
            
            features.append(np.max(rfus))
            feature_names.append(f'{locus}_max_rfu')
            
            features.append(np.std(rfus) if len(rfus) > 1 else 0.0)
            feature_names.append(f'{locus}_std_rfu')
        else:
            features.extend([0.0, 0.0, 0.0])
            feature_names.extend([f'{locus}_mean_rfu', f'{locus}_max_rfu', f'{locus}_std_rfu'])
        
        features.append(n_alleles)
        feature_names.append(f'{locus}_n_alleles')
        
        # Height ratio (2nd/1st peak)
        if len(rfus) >= 2:
            sorted_rfus = sorted(rfus, reverse=True)
            features.append(sorted_rfus[1] / sorted_rfus[0] if sorted_rfus[0] > 0 else 0.0)
        else:
            features.append(0.0)
        feature_names.append(f'{locus}_height_ratio_12')
        
        rfu_all.extend(rfus)
        allele_counts.append(n_alleles)
    
    # Global features
    if rfu_all:
        features.append(len(rfu_all))
        feature_names.append('total_peaks')
        
        features.append(np.mean(rfu_all))
        feature_names.append('mean_rfu_global')
        
        features.append(np.std(rfu_all))
        feature_names.append('std_rfu_global')
        
        features.append(stats.skew(rfu_all))
        feature_names.append('skewness_rfu_global')
        
        features.append(stats.kurtosis(rfu_all))
        feature_names.append('kurtosis_rfu_global')
    else:
        features.extend([0.0, 0.0, 0.0, 0.0, 0.0])
        feature_names.extend(['total_peaks', 'mean_rfu_global', 'std_rfu_global', 
                            'skewness_rfu_global', 'kurtosis_rfu_global'])
    
    # Anomaly indicators
    loci_with_3_peaks = sum(1 for ac in allele_counts if ac >= 3)
    loci_with_4_peaks = sum(1 for ac in allele_counts if ac >= 4)
    
    features.append(loci_with_3_peaks)
    feature_names.append('n_loci_with_3_peaks')
    
    features.append(loci_with_4_peaks)
    feature_names.append('n_loci_with_4_peaks')
    
    if allele_counts:
        features.append(np.mean(allele_counts))
        feature_names.append('mean_peaks_per_locus')
    else:
        features.append(0.0)
        feature_names.append('mean_peaks_per_locus')
    
    # Pad to 78 features
    while len(features) < 78:
        features.append(0.0)
        feature_names.append(f'padding_{len(features)}')
    
    features = features[:78]
    feature_names = feature_names[:78]
    
    return np.array(features), feature_names


def convert_directory_recursive(input_dir: str, output_dir: str):
    """
    Recursively scan input_dir for ALL CSV files and convert them.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"🔍 Scanning {input_dir} for CSV files...\n")
    
    # Recursively find all CSV files
    csv_files = list(input_path.rglob('*.csv'))
    csv_files = [f for f in csv_files if f.is_file()]
    
    print(f"📋 Found {len(csv_files)} CSV files")
    
    all_features = []
    all_labels = []
    success_count = 0
    error_count = 0
    noc_dist = {}
    
    for i, csv_file in enumerate(csv_files, 1):
        if i % 10 == 0 or i == 1:
            print(f"  Processing {i}/{len(csv_files)}...")
        
        # Extract NoC from path
        noc = extract_noc_from_path(str(csv_file))
        if noc is None:
            error_count += 1
            continue
        
        # Parse CSV
        alleles = parse_provedit_csv(str(csv_file))
        if alleles is None:
            error_count += 1
            continue
        
        # Extract features
        try:
            features, feature_names = extract_features_from_provedit(alleles)
            if features is None:
                error_count += 1
                continue
            
            all_features.append(features)
            all_labels.append(noc)
            success_count += 1
            noc_dist[noc] = noc_dist.get(noc, 0) + 1
        
        except Exception as e:
            print(f"  ⚠️  Error processing {csv_file.name}: {e}")
            error_count += 1
            continue
    
    print(f"\n✅ Conversion complete!")
    print(f"  ✓ Converted: {success_count} samples")
    print(f"  ✗ Errors: {error_count} files")
    
    if not all_features:
        print("❌ No samples converted!")
        return
    
    # Stack and save features/labels
    features_array = np.array(all_features)
    labels_array = np.array(all_labels)
    
    print(f"\n📊 Dataset shape:")
    print(f"  Features: {features_array.shape}")
    print(f"  Labels: {labels_array.shape}")
    
    # Save features
    np.save(os.path.join(output_dir, 'features.npy'), features_array)
    print(f"✓ Saved features.npy")
    
    # Save labels
    np.save(os.path.join(output_dir, 'labels.npy'), labels_array)
    print(f"✓ Saved labels.npy")
    
    # Save feature names
    with open(os.path.join(output_dir, 'feature_names.json'), 'w') as f:
        json.dump(feature_names, f, indent=2)
    print(f"✓ Saved feature_names.json ({len(feature_names)} features)")
    
    # Save metadata
    csv_files_used = [str(f.relative_to(input_path)) for f in csv_files[:10]]
    
    metadata = {
        'total_files_found': len(csv_files),
        'samples_converted': success_count,
        'conversion_errors': error_count,
        'noc_distribution': noc_dist,
        'feature_count': 78,
        'sample_files': csv_files_used
    }
    
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved metadata.json")
    
    # Save sample labels CSV
    with open(os.path.join(output_dir, 'noc_labels.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['sample_id', 'noc'])
        writer.writeheader()
        for i, label in enumerate(labels_array):
            writer.writerow({'sample_id': f'sample_{i:04d}', 'noc': label})
    print(f"✓ Saved noc_labels.csv")
    
    print(f"\n🧬 NoC Distribution:")
    for noc in sorted(noc_dist.keys()):
        count = noc_dist[noc]
        bar = "█" * count
        print(f"  NoC {noc:2d}: {count:3d} samples {bar}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert ALL PROVEDIt DNA CSVs to NoC training format (expanded)'
    )
    parser.add_argument('--input-dir', required=True, 
                       help='Input directory containing PROVEDIt CSV files')
    parser.add_argument('--output-dir', default='data/provedit_converted_all',
                       help='Output directory for feature files')
    
    args = parser.parse_args()
    
    convert_directory_recursive(args.input_dir, args.output_dir)
