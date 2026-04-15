"""Helper utility functions."""

import torch
import random
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, Any, Optional


def set_seed(seed: int = 42):
    """Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_config(config: Dict[str, Any], save_path: str):
    """Save configuration dictionary to YAML file.
    
    Args:
        config: Configuration dictionary
        save_path: Path to save the configuration file
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def load_checkpoint(checkpoint_path: str, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer] = None):
    """Load model checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optional optimizer to load state into
        
    Returns:
        Dictionary containing checkpoint information (epoch, loss, etc.)
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return {
        'epoch': checkpoint.get('epoch', 0),
        'loss': checkpoint.get('loss', None),
        'other': checkpoint.get('other', {})
    }


def get_device() -> torch.device:
    """Get the appropriate device (GPU if available, else CPU).
    
    Returns:
        PyTorch device
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
