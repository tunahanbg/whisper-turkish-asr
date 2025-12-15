"""
Test configuration management.
"""

import pytest
from config import Config, get_config


def test_config_singleton():
    """Config Singleton pattern testi."""
    config1 = Config()
    config2 = Config()
    assert config1 is config2


def test_config_get():
    """Config get method testi."""
    config = get_config()
    
    # Nested key access
    model_name = config.get('model.name')
    assert model_name is not None
    
    # Default value
    nonexistent = config.get('nonexistent.key', 'default')
    assert nonexistent == 'default'


def test_config_properties():
    """Config properties testi."""
    config = get_config()
    
    assert isinstance(config.model_config, dict)
    assert isinstance(config.audio_config, dict)
    assert isinstance(config.vad_config, dict)
    assert isinstance(config.transcription_config, dict)

