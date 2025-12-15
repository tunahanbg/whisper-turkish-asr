"""
Faster-Whisper model wrapper.

Faster-Whisper (CTranslate2) implementation for optimized inference.
- 3-4x faster than standard Whisper
- Lower memory usage
- Supports custom/fine-tuned models
- int8 quantization for speed
"""

from typing import Optional, Dict, Any, List
from pathlib import Path
import numpy as np
from loguru import logger

try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False
    logger.warning("faster-whisper not installed. Install with: pip install faster-whisper")

from config import config
from src.models.base_asr import BaseASR


class FasterWhisperASR(BaseASR):
    """
    Faster-Whisper model wrapper using CTranslate2.
    
    Features:
    - 3-4x faster than standard Whisper
    - Supports custom/fine-tuned models
    - int8 quantization for additional speed
    - Lower memory footprint
    - Same API as WhisperASR for easy swapping
    
    Fine-tuned model support:
    - Set model_path in config to use custom model
    - Compatible with Hugging Face fine-tuned Whisper models
    """
    
    def __init__(self, custom_config: Optional[Dict[str, Any]] = None):
        """
        Args:
            custom_config: Özel model config (None = global config)
        """
        if not FASTER_WHISPER_AVAILABLE:
            raise ImportError(
                "faster-whisper is not installed. "
                "Install with: pip install faster-whisper"
            )
        
        # BaseASR initialization
        model_config = custom_config or config.model_config
        super().__init__(model_config)
        
        self.transcription_config = config.transcription_config
        
        # Model parametreleri
        self.model_name = self.model_config.get('name', 'faster-whisper')
        self.variant = self.model_config.get('variant', 'medium')
        self.device = self._get_device()
        self.compute_type = self.model_config.get('compute_type', 'int8')
        
        # Custom model path (fine-tuned models için)
        self.model_path = self.model_config.get('model_path')
        if self.model_path:
            logger.info(f"Using custom model from: {self.model_path}")
        
        # Download root (pre-trained models için)
        self.download_root = self.model_config.get('download_root', './checkpoints')
        
        # Performance parameters
        self.num_workers = self.model_config.get('num_workers', 1)
        self.cpu_threads = self.model_config.get('cpu_threads', 4)
        
        logger.info(f"FasterWhisperASR initialized - Variant: {self.variant}, "
                   f"Device: {self.device}, Compute: {self.compute_type}")
    
    def _get_device(self) -> str:
        """
        Device seçimi.
        
        faster-whisper device options:
        - "cuda": NVIDIA GPU
        - "cpu": CPU
        - "auto": Otomatik seçim
        
        Note: MPS (Apple Silicon) henüz desteklenmiyor.
        """
        device_config = self.model_config.get('device', 'cpu')
        
        # MPS varsa CPU'ya fallback
        if device_config in ['mps', 'auto']:
            logger.info("MPS not supported by faster-whisper, using CPU")
            return 'cpu'
        
        # CUDA veya CPU
        if device_config in ['cuda', 'cpu']:
            return device_config
        
        logger.warning(f"Unknown device '{device_config}', using CPU")
        return 'cpu'
    
    def load(self) -> None:
        """Model'i yükle."""
        if self.is_loaded:
            logger.warning("Model already loaded")
            return
        
        try:
            logger.info("Loading Faster-Whisper model...")
            
            # Model path (custom veya pre-trained)
            model_path_or_name = self.model_path or self.variant
            
            # Model yükle
            self.model = WhisperModel(
                model_path_or_name,
                device=self.device,
                compute_type=self.compute_type,
                download_root=self.download_root if not self.model_path else None,
                num_workers=self.num_workers,
                cpu_threads=self.cpu_threads,
            )
            
            self.is_loaded = True
            logger.info(f"Model loaded successfully on {self.device} "
                       f"(compute_type: {self.compute_type})")
            
            if self.model_path:
                logger.info(f"✅ Custom model loaded from: {self.model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.is_loaded = False
            raise
    
    def transcribe(
        self,
        audio: np.ndarray,
        language: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Audio'yu metne çevir.
        
        Args:
            audio: Audio numpy array (float32, mono, 16kHz)
            language: Dil kodu (None = otomatik tespit)
            **kwargs: Override parametreleri
        
        Returns:
            {
                'text': str,
                'language': str,
                'segments': List[Dict],
                'duration': float,
            }
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        try:
            # Transcription parametrelerini hazırla
            transcribe_params = self._get_transcribe_params(language, **kwargs)
            
            logger.debug(f"Transcribing audio - Language: {language or 'auto'}, "
                        f"Length: {len(audio)/16000:.2f}s")
            
            # Transkripsiyon yap
            segments, info = self.model.transcribe(
                audio,
                **transcribe_params
            )
            
            # Segments'i list'e çevir (generator)
            segments_list = list(segments)
            
            # Metni birleştir
            text = " ".join([segment.text.strip() for segment in segments_list])
            
            # Detected language
            detected_language = info.language if hasattr(info, 'language') else language
            
            # Segment bilgilerini formatla
            formatted_segments = []
            for segment in segments_list:
                formatted_segments.append({
                    'id': segment.id,
                    'start': segment.start,
                    'end': segment.end,
                    'text': segment.text.strip(),
                    'avg_logprob': segment.avg_logprob if hasattr(segment, 'avg_logprob') else None,
                    'no_speech_prob': segment.no_speech_prob if hasattr(segment, 'no_speech_prob') else None,
                })
            
            logger.info(f"Transcription completed - Detected language: {detected_language}")
            
            # WhisperASR ile aynı format
            return {
                'text': text,
                'language': detected_language,
                'segments': formatted_segments,
                'duration': info.duration if hasattr(info, 'duration') else None,
            }
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise
    
    def _get_transcribe_params(
        self,
        language: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Transkripsiyon parametrelerini hazırla.
        Config + override parametreleri birleştir.
        """
        # Config'den default parametreler
        params = {
            'language': language or config.get('language.default'),
            'task': self.transcription_config.get('task', 'transcribe'),
            'temperature': self.transcription_config.get('temperature', 0.0),
            'beam_size': self.transcription_config.get('beam_size', 5),
            'best_of': self.transcription_config.get('best_of', 5),
            'patience': self.transcription_config.get('patience', 1.0),
            'length_penalty': self.transcription_config.get('length_penalty', 1.0),
            'compression_ratio_threshold': self.transcription_config.get('compression_ratio_threshold', 2.4),
            'no_speech_threshold': self.transcription_config.get('no_speech_threshold', 0.6),
            'condition_on_previous_text': self.transcription_config.get('condition_on_previous_text', False),
            'initial_prompt': self.transcription_config.get('initial_prompt'),
            'word_timestamps': self.transcription_config.get('word_timestamps', False),
            # Faster-whisper specific
            'vad_filter': self.transcription_config.get('vad_filter', False),
            'vad_parameters': self.transcription_config.get('vad_parameters'),
        }
        
        # Override parametreleri
        params.update(kwargs)
        
        # None değerleri temizle
        params = {k: v for k, v in params.items() if v is not None}
        
        return params
    
    def unload(self) -> None:
        """Model'i memory'den kaldır."""
        if self.is_loaded:
            del self.model
            self.model = None
            self.is_loaded = False
            logger.info("Model unloaded from memory")
    
    def __repr__(self) -> str:
        model_info = f"variant={self.variant}" if not self.model_path else f"custom={Path(self.model_path).name}"
        return (f"FasterWhisperASR({model_info}, device={self.device}, "
                f"compute={self.compute_type}, loaded={self.is_loaded})")
    
    def __del__(self):
        """Destructor - model'i temizle."""
        self.unload()

