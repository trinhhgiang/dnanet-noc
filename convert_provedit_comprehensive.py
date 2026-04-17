import os

"""
Convert ALL PROVEDIt DNA CSVs to NoC training format (COMPREHENSIVE - All Files)

Maximum coverage: Recursively finds and parses ALL CSV files using multiple NoC extraction strategies.
Robust handling of complex naming, folder structures, and edge cases.

Usage:
    python convert_provedit_comprehensive.py --input-dir <path> --output-dir <path>
"""

import argparse
import csv
import json
import os
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from scipy import stats


def extract_noc_comprehensive(filepath: str) -> Optional[int]:
    """
    Comprehensive NoC extraction using multiple strategies with fallback chain.
    
    Strategies (in priority order):
    1. Explicit folder separation (1-Person/, 2-Person/, etc.)
    2. Explicit filename suffix (_1P, _2P, etc.)
    3. Sample ID extraction and analysis
    4. Return None for genuinely ambiguous cases
    """
    path = str(filepath).lower()
    filename = os.path.basename(filepath).lower()
    
    # ===== STRATEGY 1: Folder-based extraction (highest priority) =====
    folder_patterns = {
        r'/1\\s*[-_]?person': 1, r'/1p[/\\\\s]': 1,
        r'/2\\s*[-_]?person': 2, r'/2p[/\\\\s]': 2,
        r'/3\\s*[-_]?person': 3, r'/3p[/\\\\s]': 3,
        r'/4\\s*[-_]?person': 4, r'/4p[/\\\\s]': 4,
        r'/5\\s*[-_]?person': 5, r'/5p[/\\\\s]': 5,
    }
    
    for pattern, noc in folder_patterns.items():
        if re.search(pattern, path):
            return noc
    
    # ===== STRATEGY 2: Explicit filename suffix (exact patterns) =====
    explicit_patterns = [
        (r'_1p\\.csv$', 1), (r'-1p\\.csv$', 1), (r'_1p_', 1),
        (r'_2p\\.csv$', 2), (r'-2p\\.csv$', 2), (r'_2p_', 2),
        (r'_3p\\.csv$', 3), (r'-3p\\.csv$', 3), (r'_3p_', 3),
        (r'_4p\\.csv$', 4), (r'-4p\\.csv$', 4), (r'_4p_', 4),
        (r'_5p\\.csv$', 5), (r'-5p\\.csv$', 5), (r'_5p_', 5),
    ]
    
    for pattern, noc in explicit_patterns:
        if re.search(pattern, filename):
            return noc
    
    # ===== STRATEGY 3: Ambiguous 2-5P handling - skip for now =====
    if re.search(r'[_-]2[-_]?5p', filename):
        return None
    
    # ===== STRATEGY 4: Metadata/reference files =====
    if any(x in filename for x in ['known', 'genotype', 'reference', 'metadata', 'config']):
        return None
    
    return None


def parse_provedit_csv(filepath: str) -> Optional[Dict]:
    """
    Parse PROVEDIt CSV file with STR allele calls.
    Flexible parser that handles various CSV formats.
    """
    loci_data = {}
    
    try:
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
        
        for encoding in encodings:
            try:
                loci_data = {}
                with open(filepath, 'r', encoding=encoding) as f:
                    reader = csv.DictReader(f)
                    if reader is None or reader.fieldnames is None:
                        continue
                    
                    row_count = 0
                    for row in reader:
                        if not row or len(row) < 2:
                            continue
                        
                        try:
                            keys = list(row.keys())
                            if len(keys) < 3:
                                continue
                            
                            locus = row.get(keys[0], '').strip()
                            size = row.get(keys[1], '').strip()
                            rfu_str = row.get(keys[2], '0').strip()
                            
                            if locus.lower() in ['locus', 'size', 'rfu', 'marker', 'dye']:
                                continue
                            if not locus or len(locus) < 2:
                                continue
                            if not size or size.lower() in ['size', 'rfu', 'height', 'dye']:
                                continue
                            
                            try:
                                rfu = float(rfu_str) if rfu_str and rfu_str != 'Height' else 0.0
                            except (ValueError, TypeError):
                                rfu = 0.0
                            
                            if locus not in loci_data:
                                loci_data[locus] = []
                            
                            loci_data[locus].append({'allele': size, 'rfu': rfu})
                            row_count += 1
                        
                        except (IndexError, KeyError, ValueError):
                            continue
                    
                    if row_count > 0:
                        return loci_data
            
            except Exception:
                continue
        
        return None
    
    except Exception as e:
        return None


def extract_features_from_provedit(alleles_dict: Dict) -> Optional[Tuple[np.ndarray, List[str]]]:
    """Extract 78 statistical features from allele data."""
    if not alleles_dict or len(alleles_dict) == 0:
        return None
    
    features = []
    feature_names = []
    
    priority_loci = ['AMEL', 'D3S1358', 'vWA', 'FGA', 'D5S818', 'D7S820', 'D8S1179',
                     'D13S317', 'D16S539', 'D18S51', 'D19S433', 'D21S11', 'D2S1338', 'D12S391']
    
    selected_loci = []
    for locus in priority_loci:
        if locus in alleles_dict:
            selected_loci.append(locus)
    
    for locus in sorted(alleles_dict.keys()):
        if locus not in selected_loci and len(selected_loci) < 20:
            selected_loci.append(locus)
    
    rfu_all = []
    allele_counts = []
    
    for locus in selected_loci[:16]:
        alleles = alleles_dict[locus]
        if not alleles:
            continue
        
        rfus = [a['rfu'] for a in alleles if a['rfu'] > 0]
        n_alleles = len(alleles)
        n_peaks = len(rfus)
        
        features.append(n_peaks)
        feature_names.append(f'{locus}_peak_count')
        
        if rfus:
            features.extend([np.mean(rfus), np.max(rfus), np.std(rfus) if len(rfus) > 1 else 0.0])
            feature_names.extend([f'{locus}_mean_rfu', f'{locus}_max_rfu', f'{locus}_std_rfu'])
        else:
            features.extend([0.0, 0.0, 0.0])
            feature_names.extend([f'{locus}_mean_rfu', f'{locus}_max_rfu', f'{locus}_std_rfu'])
        
        features.append(n_alleles)
        feature_names.append(f'{locus}_n_alleles')
        
        if len(rfus) >= 2:
            sorted_rfus = sorted(rfus, reverse=True)
            ratio = sorted_rfus[1] / sorted_rfus[0] if sorted_rfus[0] > 0 else 0.0
        else:
            ratio = 0.0
        features.append(ratio)
        feature_names.append(f'{locus}_height_ratio')
        
        rfu_all.extend(rfus)
        allele_counts.append(n_alleles)
    
    if rfu_all:
        features.extend([len(rfu_all), np.mean(rfu_all), np.std(rfu_all), 
                        stats.skew(rfu_all), stats.kurtosis(rfu_all)])
        feature_names.extend(['total_peaks', 'mean_rfu_global', 'std_rfu_global',
                            'skewness_rfu_global', 'kurtosis_rfu_global'])
    else:
        features.extend([0.0, 0.0, 0.0, 0.0, 0.0])
        feature_names.extend(['total_peaks', 'mean_rfu_global', 'std_rfu_global',
                            'skewness_rfu_global', 'kurtosis_rfu_global'])
    
    if allele_counts:
        loci_with_3_peaks = sum(1 for ac in allele_counts if ac >= 3)
        loci_with_4_peaks = sum(1 for ac in allele_counts if ac >= 4)
        features.extend([loci_with_3_peaks, loci_with_4_peaks, np.mean(allele_counts)])
        feature_names.extend(['n_loci_with_3_peaks', 'n_loci_with_4_peaks', 'mean_peaks_per_locus'])
    else:
        features.extend([0.0, 0.0, 0.0])
        feature_names.extend(['n_loci_with_3_peaks', 'n_loci_with_4_peaks', 'mean_peaks_per_locus'])
    
    while len(features) < 78:
        features.append(0.0)
        feature_names.append(f'padding_{len(features)}')
    
    return np.array(features[:78]), feature_names[:78]


def convert_directory_comprehensive(input_dir: str, output_dir: str):
    """COMPREHENSIVE converter: Maximize coverage using ALL available files."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    log_file = os.path.join(output_dir, 'conversion_log.txt')
    log = open(log_file, 'w')
    
    log.write(f"{'='*80}\\n")
    log.write(f"🧬 PROVEDIt COMPREHENSIVE Converter\\n")
    log.write(f"{'='*80}\\n")
    log.write(f"Input: {input_dir}\\n")
    log.write(f"{'='*80}\\n\\n")
    
    csv_files = sorted(list(input_path.rglob('*.csv')))
    csv_files = [f for f in csv_files if f.is_file()]
    
    print(f"🔍 Comprehensive scan: {input_dir}")
    print(f"📋 Found {len(csv_files)} CSV files\\n")
    
    log.write(f"Total CSV files: {len(csv_files)}\\n\\n")
    
    all_features = []
    all_labels = []
    all_filenames = []
    
    stats_dict = {
        'successfully_converted': 0,
        'parse_errors': 0,
        'no_noc_label': 0,
        'by_noc': {}
    }
    
    for idx, csv_file in enumerate(csv_files, 1):
        rel_path = str(csv_file.relative_to(input_path))
        
        if idx % 5 == 0 or idx == 1:
            print(f"  Processing {idx}/{len(csv_files)}...")
        
        noc = extract_noc_comprehensive(str(csv_file))
        
        if noc is None:
            stats_dict['no_noc_label'] += 1
            log.write(f"⚠️  NO LABEL: {rel_path}\\n")
            continue
        
        alleles = parse_provedit_csv(str(csv_file))
        
        if alleles is None:
            stats_dict['parse_errors'] += 1
            log.write(f"❌ PARSE ERROR: {rel_path}\\n")
            continue
        
        try:
            result = extract_features_from_provedit(alleles)
            if result is None:
                stats_dict['parse_errors'] += 1
                log.write(f"❌ NO FEATURES: {rel_path}\\n")
                continue
            
            features, feature_names = result
            
            all_features.append(features)
            all_labels.append(noc)
            all_filenames.append(rel_path)
            
            stats_dict['successfully_converted'] += 1
            stats_dict['by_noc'][noc] = stats_dict['by_noc'].get(noc, 0) + 1
            
            log.write(f"✅ SUCCESS (NoC={noc}): {rel_path}\\n")
        
        except Exception as e:
            stats_dict['parse_errors'] += 1
            log.write(f"❌ EXCEPTION: {rel_path}\\n")
    
    log.write(f"\\n{'='*80}\\n")
    log.write(f"📊 SUMMARY\\n")
    log.write(f"✅ Converted: {stats_dict['successfully_converted']} samples\\n")
    log.write(f"❌ Parse errors: {stats_dict['parse_errors']} files\\n")
    log.write(f"⚠️  No label: {stats_dict['no_noc_label']} files\\n")
    log.write(f"\\n🧬 Distribution:\\n")
    for noc in sorted(stats_dict['by_noc'].keys()):
        count = stats_dict['by_noc'][noc]
        log.write(f"  NoC {noc}: {count}\\n")
    log.close()
    
    print(f"\\n{'='*60}")
    print(f"✅ Conversion complete!")
    print(f"  ✅ Converted: {stats_dict['successfully_converted']} samples")
    print(f"  ❌ Parse errors: {stats_dict['parse_errors']} files")
    print(f"  ⚠️  No label: {stats_dict['no_noc_label']} files")
    print(f"{'='*60}")
    
    if not all_features:
        print("❌ No samples!")
        return
    
    features_array = np.array(all_features)
    labels_array = np.array(all_labels)
    
    print(f"\\n📦 Saving...")
    print(f"  Features: {features_array.shape}")
    print(f"  Labels: {labels_array.shape}")
    
    np.save(os.path.join(output_dir, 'features.npy'), features_array)
    np.save(os.path.join(output_dir, 'labels.npy'), labels_array)
    
    with open(os.path.join(output_dir, 'feature_names.json'), 'w') as f:
        json.dump(feature_names, f, indent=2)
    
    with open(os.path.join(output_dir, 'noc_labels.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['sample_id', 'noc', 'filename'])
        writer.writeheader()
        for i, (label, filename) in enumerate(zip(labels_array, all_filenames)):
            writer.writerow({'sample_id': f'sample_{i:04d}', 'noc': label, 'filename': filename})
    
    metadata = {
        'total_files_scanned': len(csv_files),
        'samples_successfully_converted': stats_dict['successfully_converted'],
        'parse_errors': stats_dict['parse_errors'],
        'files_with_no_noc_label': stats_dict['no_noc_label'],
        'noc_distribution': stats_dict['by_noc'],
        'feature_count': 78,
        'input_directory': input_dir,
    }
    
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\\n✅ Files saved")
    print(f"\\n🧬 Distribution:")
    for noc in sorted(stats_dict['by_noc'].keys()):
        count = stats_dict['by_noc'][noc]
        bar = "█" * count
        print(f"  NoC {noc}: {count:3d} {bar}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='COMPREHENSIVE PROVEDIt CSV to NoC converter'
    )
    parser.add_argument('--input-dir', required=True, help='Input directory')
    parser.add_argument('--output-dir', default='data/provedit_comprehensive',
                       help='Output directory')
    
    args = parser.parse_args()
    convert_directory_comprehensive(args.input_dir, args.output_dir)

with open('convert_provedit_comprehensive.py', 'w') as f:
    f.write(code)

print("✅ Created convert_provedit_comprehensive.py")