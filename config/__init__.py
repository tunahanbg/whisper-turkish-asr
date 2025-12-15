"""
Configuration management module.
Merkezi konfigürasyon yönetimi için utility'ler.
"""

from pathlib import Path
from typing import Any, Dict, Optional
import yaml
from loguru import logger


class Config:
    """
    Singleton pattern ile config yönetimi.
    YAML dosyasından konfigürasyonu yükler ve erişim sağlar.
    """
    
    _instance: Optional['Config'] = None
    _config: Optional[Dict[str, Any]] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._config is None:
            self.load_config()
    
    def load_config(self, config_path: Optional[str] = None) -> None:
        """Config dosyasını yükle."""
        if config_path is None:
            config_path = Path(__file__).parent / "config.yaml"
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self._config = yaml.safe_load(f)
            logger.info(f"Config loaded from {config_path}")
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            raise
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Nested key ile config değeri al.
        Örnek: config.get('model.variant') -> 'medium'
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """Config değerini güncelle (runtime)."""
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
        logger.debug(f"Config updated: {key} = {value}")
    
    def save(self, config_path: Optional[str] = None) -> None:
        """Config'i dosyaya kaydet."""
        if config_path is None:
            config_path = Path(__file__).parent / "config.yaml"
        
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self._config, f, default_flow_style=False, allow_unicode=True)
            logger.info(f"Config saved to {config_path}")
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
            raise
    
    @property
    def model_config(self) -> Dict[str, Any]:
        """Model konfigürasyonunu döndür."""
        return self._config.get('model', {})
    
    @property
    def audio_config(self) -> Dict[str, Any]:
        """Audio konfigürasyonunu döndür."""
        return self._config.get('audio', {})
    
    @property
    def vad_config(self) -> Dict[str, Any]:
        """VAD konfigürasyonunu döndür."""
        return self._config.get('vad', {})
    
    @property
    def transcription_config(self) -> Dict[str, Any]:
        """Transkripsiyon konfigürasyonunu döndür."""
        return self._config.get('transcription', {})
    
    def __repr__(self) -> str:
        return f"Config(loaded={self._config is not None})"


# Global config instance
config = Config()


def get_config() -> Config:
    """Global config instance'ı döndür."""
    return config


__all__ = ['Config', 'config', 'get_config']

