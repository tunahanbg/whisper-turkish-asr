"""
Model manager for handling multiple model types.
DRY: Gelecekte farklı modeller eklenebilir (faster-whisper, MLX, vb.)
"""

from typing import Optional, Dict, Any
from enum import Enum
from loguru import logger

from config import config
from .whisper_model import WhisperASR


class ModelType(Enum):
    """Desteklenen model tipleri."""
    WHISPER = "whisper"
    FASTER_WHISPER = "faster_whisper"
    MLX_WHISPER = "mlx_whisper"


class ModelManager:
    """
    Model yönetimi için factory pattern.
    Config'e göre uygun model'i yükler.
    """
    
    def __init__(self):
        """Initialize model manager."""
        self.current_model = None
        self.model_type = None
        
        logger.debug("ModelManager initialized")
    
    def load_model(
        self,
        model_type: Optional[str] = None,
        **kwargs
    ) -> Any:
        """
        Model'i yükle.
        
        Args:
            model_type: Model tipi (None = config'den al)
            **kwargs: Model'e özel parametreler
        
        Returns:
            Model instance
        """
        # Config'den model tipini al
        if model_type is None:
            model_type = config.get('model.name', 'whisper')
        
        try:
            model_type = ModelType(model_type.lower())
        except ValueError:
            logger.error(f"Unknown model type: {model_type}")
            logger.info("Falling back to Whisper")
            model_type = ModelType.WHISPER
        
        # Model'i oluştur
        if model_type == ModelType.WHISPER:
            model = self._load_whisper(**kwargs)
        elif model_type == ModelType.FASTER_WHISPER:
            model = self._load_faster_whisper(**kwargs)
        elif model_type == ModelType.MLX_WHISPER:
            model = self._load_mlx_whisper(**kwargs)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        self.current_model = model
        self.model_type = model_type
        
        logger.info(f"Model loaded: {model_type.value}")
        return model
    
    def _load_whisper(self, **kwargs) -> WhisperASR:
        """Standard Whisper model'i yükle."""
        model = WhisperASR(**kwargs)
        model.load()
        return model
    
    def _load_faster_whisper(self, **kwargs):
        """Faster-Whisper model'i yükle (gelecek için)."""
        logger.warning("Faster-Whisper not yet implemented")
        raise NotImplementedError("Faster-Whisper support coming soon")
    
    def _load_mlx_whisper(self, **kwargs):
        """MLX Whisper model'i yükle (Apple Silicon native - gelecek için)."""
        logger.warning("MLX Whisper not yet implemented")
        raise NotImplementedError("MLX Whisper support coming soon")
    
    def get_model(self) -> Any:
        """Mevcut model'i döndür."""
        if self.current_model is None:
            raise RuntimeError("No model loaded. Call load_model() first.")
        return self.current_model
    
    def unload_model(self) -> None:
        """Model'i bellekten kaldır."""
        if self.current_model is not None:
            if hasattr(self.current_model, 'unload'):
                self.current_model.unload()
            
            self.current_model = None
            self.model_type = None
            
            logger.info("Model unloaded")
    
    def reload_model(self, **kwargs) -> Any:
        """Model'i yeniden yükle."""
        self.unload_model()
        return self.load_model(**kwargs)
    
    @property
    def is_loaded(self) -> bool:
        """Model yüklü mü?"""
        return self.current_model is not None
    
    def __repr__(self) -> str:
        return f"ModelManager(type={self.model_type}, loaded={self.is_loaded})"

