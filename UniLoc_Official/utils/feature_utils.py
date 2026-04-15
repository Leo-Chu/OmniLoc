"""Utility functions for working with feature fields."""

import numpy as np
from typing import Dict, List, Optional


def concatenate_fields(fields_dict: Dict[str, np.ndarray], 
                       field_order: Optional[List[str]] = None) -> np.ndarray:
    """Concatenate separate feature fields into a single feature vector.
    
    Args:
        fields_dict: Dictionary with field names as keys and numpy arrays as values
        field_order: Order of fields for concatenation. If None, uses default order.
        
    Returns:
        Concatenated feature array
    """
    if field_order is None:
        field_order = ['Blk_APVec_1', 'Blk_APVec_2', 'Blk_CSI', 'Blk_RSSI', 'Blk_SNR']
    
    # Filter to only include fields that exist
    field_order = [f for f in field_order if f in fields_dict]
    
    if len(field_order) == 0:
        raise ValueError("No valid fields found in fields_dict")
    
    # Concatenate along the feature dimension (axis=1 for 2D arrays, axis=0 for 1D)
    feature_list = []
    for field in field_order:
        field_data = fields_dict[field]
        if len(field_data.shape) == 1:
            # Single sample
            feature_list.append(field_data)
        else:
            # Multiple samples - concatenate along feature dimension
            feature_list.append(field_data)
    
    if len(feature_list[0].shape) == 1:
        # Single sample case
        return np.concatenate(feature_list)
    else:
        # Multiple samples case
        return np.concatenate(feature_list, axis=1)


def get_field_info(fields_dict: Dict[str, np.ndarray]) -> Dict[str, dict]:
    """Get information about each field.
    
    Args:
        fields_dict: Dictionary with field names as keys and numpy arrays as values
        
    Returns:
        Dictionary mapping field names to their info (shape, dtype, etc.)
    """
    info = {}
    for field_name, field_data in fields_dict.items():
        info[field_name] = {
            'shape': field_data.shape,
            'dtype': field_data.dtype,
            'num_samples': field_data.shape[0] if len(field_data.shape) > 0 else 1,
            'feature_dim': field_data.shape[1] if len(field_data.shape) > 1 else field_data.shape[0]
        }
    return info
