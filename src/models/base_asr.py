"""
Base ASR Model Interface.

Abstract base class for all ASR models.
Modular design for easy extension and custom model integration.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np


class BaseASR(ABC):
    """
    Abstract base class for ASR models.
    
    All ASR implementations (Whisper, Faster-Whisper, Custom models)
    must inherit from this class and implement the required methods.
    
    This ensures:
    - Consistent interface across different models
    - Easy switching between models
    - Simple integration of fine-tuned models
    """
    
    def __init__(self, model_config: Dict[str, Any]):
        """
        Initialize base ASR model.
        
        Args:
            model_config: Model configuration dictionary
        """
        self.model_config = model_config
        self.model = None
        self._is_loaded = False
    
    @abstractmethod
    def load(self) -> None:
        """
        Load the model.
        
        Must be implemented by subclasses.
        Should set self._is_loaded = True when successful.
        """
        pass
    
    @abstractmethod
    def transcribe(
        self,
        audio: np.ndarray,
        language: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Transcribe audio to text.
        
        Args:
            audio: Audio data as numpy array (float32, mono, 16kHz)
            language: Language code (e.g., "tr", "en"). None for auto-detect.
            **kwargs: Additional transcription parameters
        
        Returns:
            Dictionary containing:
                - text: Transcribed text
                - language: Detected/specified language
                - segments: List of segments (if available)
                - other model-specific metadata
        
        Must be implemented by subclasses.
        """
        pass
    
    @abstractmethod
    def unload(self) -> None:
        """
        Unload the model from memory.
        
        Must be implemented by subclasses.
        Should set self._is_loaded = False.
        """
        pass
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded
    
    def is_model_loaded(self) -> bool:
        """Check if model is loaded (deprecated, use is_loaded property)."""
        return self._is_loaded
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information.
        
        Returns:
            Dictionary with model metadata
        """
        return {
            'name': self.__class__.__name__,
            'config': self.model_config,
            'loaded': self._is_loaded,
        }
    
    def format_output(
        self,
        result: Dict[str, Any],
        include_timestamps: bool = True,
        include_segments: bool = False,
    ) -> Dict[str, Any]:
        """
        Format transcription output to a consistent structure.
        
        Default implementation works for both Whisper and Faster-Whisper.
        Subclasses can override if needed.
        
        Args:
            result: Raw transcription result
            include_timestamps: Include timestamp information
            include_segments: Include segment details
        
        Returns:
            Formatted output dictionary
        """
        output = {
            'text': result.get('text', '').strip(),
            'language': result.get('language', 'unknown'),
        }
        
        if include_segments and 'segments' in result:
            segments = []
            for seg in result['segments']:
                segment_info = {
                    'text': seg.get('text', '').strip(),
                }
                
                if include_timestamps:
                    segment_info.update({
                        'start': seg.get('start', 0.0),
                        'end': seg.get('end', 0.0),
                    })
                
                segments.append(segment_info)
            
            output['segments'] = segments
        
        return output
    
    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}("
                f"loaded={self._is_loaded}, "
                f"config={self.model_config})")

