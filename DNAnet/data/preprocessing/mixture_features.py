"""
Mixture-specific feature extraction for NoC (Number of Contributors) prediction.

Extracts statistical features from multi-contributor DNA electropherograms:
- Peak-level statistics (count, heights, ratios)
- Locus-wise features (variance, density)
- Global mixture complexity indicators
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy import signal, stats


def extract_peaks(
    scanpoints: np.ndarray,
    threshold: float = 50.0,
    min_height: Optional[float] = None,
    prominence: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect peaks in a 1D electropherogram trace.
    
    Args:
        scanpoints: 1D array of RFU (peak heights)
        threshold: Minimum RFU to consider as signal
        min_height: Minimum peak height (default: threshold)
        prominence: Minimum peak prominence (default: None)
    
    Returns:
        (peak_indices, peak_heights)
    """
    if min_height is None:
        min_height = threshold
    
    # Find peaks using scipy
    peaks, properties = signal.find_peaks(
        scanpoints,
        height=min_height,
        prominence=prominence
    )
    
    if len(peaks) == 0:
        return np.array([], dtype=int), np.array([], dtype=float)
    
    peak_heights = scanpoints[peaks]
    return peaks, peak_heights


def extract_mixture_features(
    hidimage,
    panel,
    peak_threshold: float = 50.0,
    n_top_loci: int = 17
) -> Dict[str, float]:
    """
    Extract mixture-specific features from a DNA electropherogram for NoC prediction.
    
    Args:
        hidimage: HIDImage object containing electropherogram data
        panel: Panel object with locus/dye mappings
        peak_threshold: Minimum RFU for peak detection
        n_top_loci: Number of autosomal loci to include
    
    Returns:
        Dictionary of mixture features (~100 features total)
    """
    features = {}
    
    # Get scanpoint data (shape: n_dyes x n_scanpoints)
    data = hidimage.data if hasattr(hidimage, 'data') else hidimage._read()
    if data.ndim == 1:
        data = data[np.newaxis, :]
    
    # Get locus names (exclude non-autosomal: AMEL, DYS, etc.)
    locus_names = [
        locus for locus in panel.markers.keys()
        if not (locus.startswith('DYS') or locus.startswith('AMEL') or locus == 'TPOX')
    ][:n_top_loci]
    
    all_peak_heights = []
    
    # ========== Per-Locus Features ==========
    for i, locus_name in enumerate(locus_names):
        if i >= data.shape[0]:
            break
        
        scanpoints = data[i]
        peaks_idx, peak_heights = extract_peaks(scanpoints, threshold=peak_threshold)
        
        # 1. Count features
        n_peaks = len(peak_heights)
        features[f'{locus_name}_num_peaks'] = float(n_peaks)
        all_peak_heights.extend(peak_heights)
        
        # 2. Height statistics
        if n_peaks > 0:
            features[f'{locus_name}_max_rfu'] = float(np.max(peak_heights))
            features[f'{locus_name}_mean_rfu'] = float(np.mean(peak_heights))
            features[f'{locus_name}_min_rfu'] = float(np.min(peak_heights))
            features[f'{locus_name}_rfu_std'] = float(np.std(peak_heights))
        else:
            features[f'{locus_name}_max_rfu'] = 0.0
            features[f'{locus_name}_mean_rfu'] = 0.0
            features[f'{locus_name}_min_rfu'] = 0.0
            features[f'{locus_name}_rfu_std'] = 0.0
        
        # 3. Peak height ratios
        if n_peaks >= 2:
            sorted_heights = sorted(peak_heights, reverse=True)
            features[f'{locus_name}_height_ratio_12'] = float(sorted_heights[0] / (sorted_heights[1] + 1e-6))
            if n_peaks >= 3:
                features[f'{locus_name}_height_ratio_23'] = float(sorted_heights[1] / (sorted_heights[2] + 1e-6))
            if n_peaks >= 4:
                features[f'{locus_name}_height_ratio_34'] = float(sorted_heights[2] / (sorted_heights[3] + 1e-6))
        else:
            features[f'{locus_name}_height_ratio_12'] = 0.0
            features[f'{locus_name}_height_ratio_23'] = 0.0
            features[f'{locus_name}_height_ratio_34'] = 0.0
        
        # 4. Peak density (peaks per allele in panel)
        n_alleles = len(panel.markers[locus_name].alleles) if locus_name in panel.markers else 10
        features[f'{locus_name}_peak_density'] = float(n_peaks / max(n_alleles, 1))
        
        # 5. Position spread (indicates crowding/overlap)
        if n_peaks >= 2:
            peak_positions = peaks_idx[np.argsort(peak_heights)[-min(3, n_peaks):]]  # Top 3 peaks
            features[f'{locus_name}_position_spread'] = float(np.max(peak_positions) - np.min(peak_positions))
            features[f'{locus_name}_position_stddev'] = float(np.std(peak_positions))
        else:
            features[f'{locus_name}_position_spread'] = 0.0
            features[f'{locus_name}_position_stddev'] = 0.0
        
        # 6. Sorted RFU pattern (capture shape of peak cluster)
        if n_peaks > 0:
            sorted_rfu = sorted(peak_heights, reverse=True)
            # Log ratios to capture exponential differences
            for j in range(min(4, len(sorted_rfu))):
                features[f'{locus_name}_ranked_rfu_{j}'] = float(sorted_rfu[j])
        else:
            for j in range(4):
                features[f'{locus_name}_ranked_rfu_{j}'] = 0.0
    
    # ========== Global Features ==========
    
    # 1. Total peaks and statistics across all loci
    features['total_peaks'] = float(len(all_peak_heights))
    features['total_peaks_normalized'] = float(len(all_peak_heights) / len(locus_names))
    
    if all_peak_heights:
        all_heights_array = np.array(all_peak_heights)
        features['global_mean_rfu'] = float(np.mean(all_heights_array))
        features['global_std_rfu'] = float(np.std(all_heights_array))
        features['global_max_rfu'] = float(np.max(all_heights_array))
        features['global_min_rfu'] = float(np.min(all_heights_array))
        features['global_rfu_cv'] = float(np.std(all_heights_array) / (np.mean(all_heights_array) + 1e-6))
        
        # 2. Distribution shape (skewness, kurtosis)
        features['global_rfu_skewness'] = float(stats.skew(all_heights_array))
        features['global_rfu_kurtosis'] = float(stats.kurtosis(all_heights_array))
        
        # 3. Entropy and information content
        normalized_heights = all_heights_array / (np.sum(all_heights_array) + 1e-6)
        features['global_rfu_entropy'] = float(stats.entropy(normalized_heights))
        
        # 4. Distribution quantiles
        features['global_rfu_q25'] = float(np.percentile(all_heights_array, 25))
        features['global_rfu_q50'] = float(np.percentile(all_heights_array, 50))
        features['global_rfu_q75'] = float(np.percentile(all_heights_array, 75))
    else:
        features['global_mean_rfu'] = 0.0
        features['global_std_rfu'] = 0.0
        features['global_max_rfu'] = 0.0
        features['global_min_rfu'] = 0.0
        features['global_rfu_cv'] = 0.0
        features['global_rfu_skewness'] = 0.0
        features['global_rfu_kurtosis'] = 0.0
        features['global_rfu_entropy'] = 0.0
        features['global_rfu_q25'] = 0.0
        features['global_rfu_q50'] = 0.0
        features['global_rfu_q75'] = 0.0
    
    # 5. Locus-wise peak pattern anomalies
    peak_counts = np.array([features.get(f'{loc}_num_peaks', 0) for loc in locus_names])
    features['n_loci_with_4_peaks'] = float(np.sum(peak_counts >= 4))
    features['n_loci_with_3_peaks'] = float(np.sum(peak_counts >= 3))
    features['n_loci_with_2_peaks'] = float(np.sum(peak_counts >= 2))
    features['max_peaks_any_locus'] = float(np.max(peak_counts))
    features['mean_peaks_per_locus'] = float(np.mean(peak_counts))
    features['std_peaks_per_locus'] = float(np.std(peak_counts))
    
    # 6. RFU homogeneity (indicator of mixing balance)
    mean_rfus = np.array([features.get(f'{loc}_mean_rfu', 0) for loc in locus_names])
    features['mean_rfu_across_loci_std'] = float(np.std(mean_rfus[mean_rfus > 0]))
    
    return features


def normalize_features(features: Dict[str, float]) -> Dict[str, float]:
    """
    Normalize features to zero mean, unit variance (for ML models).
    Requires pre-computed statistics from training set.
    
    Args:
        features: Raw feature dictionary
    
    Returns:
        Normalized feature dictionary
    """
    # Placeholder: actual normalization would use training set stats
    # For now, return as-is; normalization applied in dataset class
    return features
