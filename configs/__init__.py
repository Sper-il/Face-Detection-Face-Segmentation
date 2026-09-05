"""
Configuration package for Face Detection & Segmentation project.
Provides YAML-based configuration loading.
"""

import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load YAML configuration file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Dictionary containing configuration
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def load_detection_config() -> Dict[str, Any]:
    """Load detection configuration."""
    config_dir = Path(__file__).parent
    return load_config(config_dir / 'detection_config.yaml')


def load_segmentation_config() -> Dict[str, Any]:
    """Load segmentation configuration."""
    config_dir = Path(__file__).parent
    return load_config(config_dir / 'segmentation_config.yaml')


__all__ = [
    'load_config',
    'load_detection_config',
    'load_segmentation_config'
]
