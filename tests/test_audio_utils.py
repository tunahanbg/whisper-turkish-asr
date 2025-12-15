"""
Test audio utility functions.
"""

import pytest
import numpy as np
from pathlib import Path

from src.utils.audio_utils import (
    validate_audio,
    get_audio_duration,
    format_timestamp,
)


def test_validate_audio():
    """Audio validation testi."""
    # Valid audio
    audio = np.random.randn(16000)  # 1 saniye @ 16kHz
    assert validate_audio(audio, 16000, min_duration=0.5)
    
    # Too short
    short_audio = np.random.randn(100)
    assert not validate_audio(short_audio, 16000, min_duration=1.0)
    
    # Silent audio
    silent_audio = np.zeros(16000)
    assert not validate_audio(silent_audio, 16000)


def test_get_audio_duration():
    """Audio duration hesaplama testi."""
    audio = np.random.randn(48000)  # 3 saniye @ 16kHz
    duration = get_audio_duration(audio, sample_rate=16000)
    assert duration == 3.0


def test_format_timestamp():
    """Timestamp formatlamatesti."""
    # 1 dakika 30 saniye
    timestamp = format_timestamp(90.5)
    assert timestamp == "01:30.500"
    
    # 1 saat 5 dakika 30 saniye
    timestamp = format_timestamp(3930.123)
    assert timestamp == "01:05:30.123"

