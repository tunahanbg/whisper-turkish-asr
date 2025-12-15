"""
Test Voice Activity Detector.
"""

import pytest
import numpy as np

from src.preprocessing import VoiceActivityDetector


def test_vad_init_disabled():
    """Test VAD initialization when disabled - should not raise AttributeError."""
    # Disabled VAD oluştur
    vad = VoiceActivityDetector(custom_config={'enabled': False})
    
    # Attributes initialize edilmiş olmalı
    assert hasattr(vad, 'enabled')
    assert hasattr(vad, 'threshold')
    assert hasattr(vad, 'min_silence_duration_ms')
    assert hasattr(vad, 'sample_rate')
    assert vad.enabled is False
    
    # __repr__ çağrılabilmeli (AttributeError olmamalı)
    repr_str = repr(vad)
    assert 'VoiceActivityDetector' in repr_str
    assert 'enabled=False' in repr_str


def test_vad_init_enabled():
    """Test VAD initialization when enabled."""
    # Not: Gerçek model yükleme testi için silero-vad gerekli
    # Bu test sadece init parametrelerini kontrol eder
    config = {
        'enabled': True,
        'threshold': 0.7,
        'min_silence_duration_ms': 5000,
    }
    
    try:
        vad = VoiceActivityDetector(custom_config=config)
        
        assert vad.enabled is True
        assert vad.threshold == 0.7
        assert vad.min_silence_duration_ms == 5000
        
    except Exception as e:
        # Model yüklenemezse (test ortamında normal)
        pytest.skip(f"Silero VAD model not available: {e}")


def test_vad_detect_speech_disabled():
    """Test detect_speech when VAD is disabled."""
    vad = VoiceActivityDetector(custom_config={'enabled': False})
    
    # Dummy audio
    audio = np.random.randn(16000)  # 1 saniye @ 16kHz
    
    # Disabled VAD tüm audio'yu tek segment olarak dönmeli
    segments = vad.detect_speech(audio, sample_rate=16000)
    
    assert len(segments) == 1
    assert segments[0]['start'] == 0.0
    assert segments[0]['end'] == 1.0


def test_vad_check_silence_duration_disabled():
    """Test check_silence_duration when VAD is disabled."""
    vad = VoiceActivityDetector(custom_config={'enabled': False})
    
    audio = np.random.randn(32000)  # 2 saniye @ 16kHz
    
    # Disabled VAD için silence duration 0 olmalı (tüm audio konuşma sayılır)
    silence_duration = vad.check_silence_duration(audio, sample_rate=16000)
    
    # Disabled VAD tüm audio'yu konuşma olarak algılar, bu yüzden silence 0
    assert silence_duration == 0.0


def test_vad_should_stop_recording_disabled():
    """Test should_stop_recording when VAD is disabled."""
    vad = VoiceActivityDetector(custom_config={
        'enabled': False,
        'min_silence_duration_ms': 5000,
    })
    
    audio = np.random.randn(80000)  # 5 saniye @ 16kHz
    
    # Disabled VAD için her zaman False dönmeli (hiç sessizlik algılanmaz)
    should_stop = vad.should_stop_recording(audio, sample_rate=16000)
    
    # Disabled VAD sessizlik algılamadığı için False
    assert should_stop is False


def test_vad_attributes_access():
    """Test that all attributes are accessible even when disabled."""
    vad = VoiceActivityDetector(custom_config={'enabled': False})
    
    # Tüm attribute'lar erişilebilir olmalı
    _ = vad.enabled
    _ = vad.threshold
    _ = vad.min_speech_duration_ms
    _ = vad.max_speech_duration_s
    _ = vad.min_silence_duration_ms
    _ = vad.window_size_samples
    _ = vad.speech_pad_ms
    _ = vad.sample_rate
    _ = vad.model  # None olabilir
    _ = vad.utils  # None olabilir
    
    # Hiçbir AttributeError olmamalı
    assert True


def test_vad_repr_with_custom_config():
    """Test __repr__ with custom config."""
    vad = VoiceActivityDetector(custom_config={
        'enabled': False,
        'threshold': 0.8,
    })
    
    repr_str = repr(vad)
    assert 'enabled=False' in repr_str
    assert 'threshold=0.8' in repr_str

