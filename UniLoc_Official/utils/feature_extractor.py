"""Utility to extract individual feature fields from concatenated feature vectors.

NOTE: The current data format stores fields separately and Blk_CSI uses ONLY magnitude
(no phase). This utility is for legacy concatenated format. For the new format, access
fields directly from the pickle files.
"""

import numpy as np
from typing import Dict, Tuple


class FeatureExtractor:
    """Extract individual feature fields from concatenated feature vectors.
    
    NOTE: This is for legacy concatenated format. Current format stores fields separately.
    Blk_CSI uses ONLY magnitude (no phase).
    
    The features are concatenated in this order:
    1. Blk_APVec_1
    2. Blk_APVec_2
    3. Blk_CSI (magnitude only - phase NOT used)
    4. Blk_RSSI
    5. Blk_SNR
    """
    
    def __init__(self):
        """Initialize feature extractor with dimension information.
        
        Based on the maximum dimensions found across all files:
        - Blk_APVec_1: max 222 dims (from Builing7_Floor2/3)
        - Blk_APVec_2: 3317 dims (constant)
        - Blk_CSI: max 52*222 = 11544 dims (magnitude) + 11544 dims (phase) = 23088 dims
        - Blk_RSSI: max 222 dims
        - Blk_SNR: max 222 dims
        Total: 222 + 3317 + 23088 + 222 + 222 = 27071 dims
        """
        # Maximum dimensions found across all files
        self.dims = {
            'Blk_APVec_1': 222,
            'Blk_APVec_2': 3317,
            'Blk_CSI_magnitude': 11544,  # 52 * 222
            'Blk_CSI_phase': 11544,      # 52 * 222
            'Blk_RSSI': 222,
            'Blk_SNR': 222
        }
        
        # Calculate start and end indices for each field
        self.indices = {}
        start_idx = 0
        
        # Blk_APVec_1
        self.indices['Blk_APVec_1'] = (start_idx, start_idx + self.dims['Blk_APVec_1'])
        start_idx += self.dims['Blk_APVec_1']
        
        # Blk_APVec_2
        self.indices['Blk_APVec_2'] = (start_idx, start_idx + self.dims['Blk_APVec_2'])
        start_idx += self.dims['Blk_APVec_2']
        
        # Blk_CSI (magnitude then phase)
        self.indices['Blk_CSI_magnitude'] = (start_idx, start_idx + self.dims['Blk_CSI_magnitude'])
        start_idx += self.dims['Blk_CSI_magnitude']
        self.indices['Blk_CSI_phase'] = (start_idx, start_idx + self.dims['Blk_CSI_phase'])
        start_idx += self.dims['Blk_CSI_phase']
        
        # Blk_RSSI
        self.indices['Blk_RSSI'] = (start_idx, start_idx + self.dims['Blk_RSSI'])
        start_idx += self.dims['Blk_RSSI']
        
        # Blk_SNR
        self.indices['Blk_SNR'] = (start_idx, start_idx + self.dims['Blk_SNR'])
        start_idx += self.dims['Blk_SNR']
        
        self.total_dims = start_idx
        assert self.total_dims == 27071, f"Total dimensions should be 27071, got {self.total_dims}"
    
    def extract_field(self, features: np.ndarray, field_name: str) -> np.ndarray:
        """Extract a specific field from concatenated features.
        
        Args:
            features: Feature array of shape (N, 27071) or (27071,)
            field_name: Name of field to extract. Options:
                - 'Blk_APVec_1'
                - 'Blk_APVec_2'
                - 'Blk_CSI_magnitude'
                - 'Blk_CSI_phase'
                - 'Blk_CSI' (returns both magnitude and phase concatenated)
                - 'Blk_RSSI'
                - 'Blk_SNR'
        
        Returns:
            Extracted field array
        """
        if field_name not in self.indices and field_name != 'Blk_CSI':
            raise ValueError(f"Unknown field: {field_name}. Available: {list(self.indices.keys()) + ['Blk_CSI']}")
        
        # Handle Blk_CSI special case
        if field_name == 'Blk_CSI':
            magnitude = self.extract_field(features, 'Blk_CSI_magnitude')
            phase = self.extract_field(features, 'Blk_CSI_phase')
            return np.concatenate([magnitude, phase], axis=-1 if len(features.shape) > 1 else 0)
        
        start_idx, end_idx = self.indices[field_name]
        
        if len(features.shape) == 1:
            # Single sample
            return features[start_idx:end_idx]
        else:
            # Multiple samples
            return features[:, start_idx:end_idx]
    
    def extract_all_fields(self, features: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract all fields from concatenated features.
        
        Args:
            features: Feature array of shape (N, 27071) or (27071,)
        
        Returns:
            Dictionary mapping field names to extracted arrays
        """
        return {
            'Blk_APVec_1': self.extract_field(features, 'Blk_APVec_1'),
            'Blk_APVec_2': self.extract_field(features, 'Blk_APVec_2'),
            'Blk_CSI_magnitude': self.extract_field(features, 'Blk_CSI_magnitude'),
            'Blk_CSI_phase': self.extract_field(features, 'Blk_CSI_phase'),
            'Blk_RSSI': self.extract_field(features, 'Blk_RSSI'),
            'Blk_SNR': self.extract_field(features, 'Blk_SNR')
        }
    
    def get_field_info(self) -> Dict[str, Tuple[int, int, int]]:
        """Get information about each field.
        
        Returns:
            Dictionary mapping field names to (start_idx, end_idx, num_dims)
        """
        info = {}
        for field_name, (start, end) in self.indices.items():
            info[field_name] = (start, end, end - start)
        return info
    
    def print_feature_structure(self):
        """Print the structure of the concatenated feature vector."""
        print("=" * 60)
        print("FEATURE VECTOR STRUCTURE (Total: 27,071 dimensions)")
        print("=" * 60)
        print(f"{'Field':<25} {'Start':<10} {'End':<10} {'Dimensions':<12}")
        print("-" * 60)
        
        for field_name, (start, end) in self.indices.items():
            dims = end - start
            print(f"{field_name:<25} {start:<10} {end:<10} {dims:<12}")
        
        # Also show combined Blk_CSI
        csi_start = self.indices['Blk_CSI_magnitude'][0]
        csi_end = self.indices['Blk_CSI_phase'][1]
        csi_dims = csi_end - csi_start
        print(f"{'Blk_CSI (combined)':<25} {csi_start:<10} {csi_end:<10} {csi_dims:<12}")
        print("=" * 60)
        print(f"\nTotal dimensions: {self.total_dims}")
        print("\nNote: Blk_CSI is stored as magnitude (first half) and phase (second half)")
        print("      To reconstruct complex numbers: magnitude * exp(1j * phase)")


def reconstruct_complex_csi(magnitude: np.ndarray, phase: np.ndarray) -> np.ndarray:
    """Reconstruct complex CSI from magnitude and phase.
    
    Args:
        magnitude: Magnitude array
        phase: Phase array (in radians)
    
    Returns:
        Complex array: magnitude * exp(1j * phase)
    """
    return magnitude * np.exp(1j * phase)


if __name__ == '__main__':
    # Example usage
    extractor = FeatureExtractor()
    extractor.print_feature_structure()
    
    # Example: Load and extract features
    print("\n" + "=" * 60)
    print("EXAMPLE USAGE")
    print("=" * 60)
    print("""
    import pickle
    import numpy as np
    from utils.feature_extractor import FeatureExtractor
    
    # Load data
    with open('data/training.pkl', 'rb') as f:
        data = pickle.load(f)
    
    features = data['features']  # Shape: (N, 27071)
    
    # Create extractor
    extractor = FeatureExtractor()
    
    # Extract individual fields
    apvec_1 = extractor.extract_field(features, 'Blk_APVec_1')  # Shape: (N, 222)
    apvec_2 = extractor.extract_field(features, 'Blk_APVec_2')  # Shape: (N, 3317)
    csi_mag = extractor.extract_field(features, 'Blk_CSI_magnitude')  # Shape: (N, 11544)
    csi_phase = extractor.extract_field(features, 'Blk_CSI_phase')  # Shape: (N, 11544)
    rssi = extractor.extract_field(features, 'Blk_RSSI')  # Shape: (N, 222)
    snr = extractor.extract_field(features, 'Blk_SNR')  # Shape: (N, 222)
    
    # Or extract all at once
    all_fields = extractor.extract_all_fields(features)
    """)
