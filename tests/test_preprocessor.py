"""
Test audio preprocessor.
"""

import pytest
import numpy as np

from src.preprocessing import AudioPreprocessor


def test_preprocessor_init():
    """Preprocessor initialization testi."""
    preprocessor = AudioPreprocessor()
    assert preprocessor is not None


def test_preprocessor_normalize():
    """Audio normalization testi."""
    preprocessor = AudioPreprocessor({
        'enabled': True,
        'normalize': True,
        'trim_silence': False,
        'denoise': False,
    })
    
    # Audio oluştur
    audio = np.random.randn(16000) * 0.5
    
    # Process
    processed = preprocessor.process(audio)
    
    # Normalized audio max value ~1.0 olmalı
    assert np.abs(processed).max() <= 1.0


def test_preprocessor_disabled():
    """Disabled preprocessor testi."""
    preprocessor = AudioPreprocessor({'enabled': False})
    
    audio = np.random.randn(16000)
    processed = preprocessor.process(audio)
    
    # Disabled ise orijinal audio dönmeli
    assert np.array_equal(audio, processed)

