"""
ASR System - Automatic Speech Recognition
Türkçe ve İngilizce destekli yerel konuşma tanıma sistemi.
"""

__version__ = "0.1.0"
__author__ = "Tunahan Başaran Güneysu"
__email__ = "tunahanbg@gmail.com"

from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

# Directories
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "checkpoints"
LOGS_DIR = PROJECT_ROOT / "logs"
CONFIG_DIR = PROJECT_ROOT / "config"

# Ensure directories exist
for directory in [DATA_DIR, MODELS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

__all__ = [
    '__version__',
    'PROJECT_ROOT',
    'DATA_DIR',
    'MODELS_DIR',
    'LOGS_DIR',
    'CONFIG_DIR',
]

