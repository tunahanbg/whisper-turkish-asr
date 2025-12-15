"""
Model management module.
DRY: Model yükleme ve inference için merkezi modül.
"""

from .whisper_model import WhisperASR
from .model_manager import ModelManager

__all__ = ['WhisperASR', 'ModelManager']

