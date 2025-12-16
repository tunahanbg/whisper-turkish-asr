"""
Whisper model wrapper.
DRY: Config-driven, modüler Whisper interface.
Supports both standard Whisper and custom HuggingFace models (including quantized).
"""

from typing import Optional, Dict, Any, List
from pathlib import Path
import numpy as np
import torch
import whisper
from loguru import logger

from config import config

# HuggingFace transformers için (custom/quantized models)
try:
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("transformers not installed. Custom model support disabled.")


class WhisperASR:
    """
    Whisper model wrapper.
    Config'den tüm parametreleri alır, modelin değişmesi için sadece config güncellenir.
    
    Features:
    - Standard Whisper models (openai/whisper)
    - Custom HuggingFace models (fine-tuned, quantized)
    - Automatic model type detection
    """
    
    def __init__(self, custom_config: Optional[Dict[str, Any]] = None):
        """
        Args:
            custom_config: Özel model config (None = global config)
        """
        self.model_config = custom_config or config.model_config
        self.transcription_config = config.transcription_config
        
        # Model parametreleri
        self.model_name = self.model_config.get('name', 'whisper')
        self.variant = self.model_config.get('variant', 'medium')
        self.device = self._get_device()
        self.compute_type = self.model_config.get('compute_type', 'float16')
        self.download_root = self.model_config.get('download_root', './checkpoints')
        
        # Custom model path (HuggingFace ID or local path)
        self.model_path = self.model_config.get('model_path')
        self.use_transformers = bool(self.model_path)
        
        # Model henüz yüklenmedi
        self.model = None
        self.processor = None  # For transformers models
        self._is_loaded = False
        
        if self.model_path:
            logger.info(f"WhisperASR initialized - Custom model: {self.model_path}, Device: {self.device}")
        else:
            logger.info(f"WhisperASR initialized - Variant: {self.variant}, Device: {self.device}")
    
    def _get_device(self) -> str:
        """En uygun device'ı belirle."""
        device_config = self.model_config.get('device', 'mps')
        
        # MPS (Apple Silicon) kontrolü
        if device_config == 'mps' and torch.backends.mps.is_available():
            return 'mps'
        
        # CUDA kontrolü
        if device_config == 'cuda' and torch.cuda.is_available():
            return 'cuda'
        
        # Fallback to CPU
        logger.warning(f"Requested device '{device_config}' not available, using CPU")
        return 'cpu'
    
    def load(self) -> None:
        """Model'i yükle."""
        if self._is_loaded:
            logger.debug("Model already loaded")
            return
        
        try:
            if self.use_transformers:
                self._load_transformers_model()
            else:
                self._load_standard_model()
            
            self._is_loaded = True
            logger.info(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _load_standard_model(self) -> None:
        """Load standard OpenAI Whisper model."""
        logger.info(f"Loading Whisper {self.variant} model...")
        
        # Download root dizinini oluştur
        Path(self.download_root).mkdir(parents=True, exist_ok=True)
        
        # Model'i yükle
        self.model = whisper.load_model(
            self.variant,
            device=self.device,
            download_root=self.download_root,
        )
    
    def _load_transformers_model(self) -> None:
        """Load custom HuggingFace model (supports quantized models)."""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers is required for custom models. "
                "Install with: pip install transformers accelerate"
            )
        
        logger.info(f"Loading HuggingFace model from: {self.model_path}")
        
        # Determine torch dtype
        torch_dtype = torch.float16 if self.compute_type in ['float16', 'fp16'] else torch.float32
        
        # Load processor (tokenizer + feature extractor)
        self.processor = WhisperProcessor.from_pretrained(self.model_path)
        
        # Load model (handles quantization automatically)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch_dtype,
            device_map=self.device if self.device != 'mps' else 'cpu',  # MPS not fully supported
            low_cpu_mem_usage=True,
        )
        
        # Move to device if not using device_map
        if self.device != 'cuda':
            self.model = self.model.to(self.device)
        
        # BetterTransformer optimization (inference speedup)
        try:
            self.model = self.model.to_bettertransformer()
            logger.info("✅ BetterTransformer enabled (faster inference)")
        except Exception as e:
            logger.debug(f"BetterTransformer not available: {e}")
        
        # Torch compile (PyTorch 2.0+)
        if hasattr(torch, 'compile') and self.device in ['cuda', 'mps']:
            try:
                self.model = torch.compile(self.model, mode='reduce-overhead')
                logger.info("✅ Torch compile enabled (PyTorch 2.0+ optimization)")
            except Exception as e:
                logger.debug(f"Torch compile not available: {e}")
        
        logger.info(f"✅ Custom model loaded: {self.model_path}")
        logger.info(f"   - Device: {self.device}")
        logger.info(f"   - Torch dtype: {torch_dtype}")
        
        # Check if quantized
        if hasattr(self.model.config, 'quantization_config'):
            quant_config = self.model.config.quantization_config
            try:
                # Try dict-like access
                if isinstance(quant_config, dict):
                    logger.info(f"   - Quantization: {quant_config.get('quant_method', 'unknown')}")
                else:
                    # Try object attribute access
                    if hasattr(quant_config, 'quant_method'):
                        logger.info(f"   - Quantization: {quant_config.quant_method}")
                    if hasattr(quant_config, 'config_groups'):
                        logger.info(f"   - Quantization config detected")
            except Exception as e:
                logger.debug(f"Could not read quantization details: {e}")
    
    def transcribe(
        self,
        audio: np.ndarray,
        language: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Ses verisini metne çevir.
        
        Args:
            audio: Audio numpy array
            language: Dil kodu ('tr', 'en', None=auto)
            **kwargs: Whisper transcribe için ek parametreler
        
        Returns:
            Transcription result dictionary
        """
        if not self._is_loaded:
            self.load()
        
        try:
            if self.use_transformers:
                return self._transcribe_transformers(audio, language, **kwargs)
            else:
                return self._transcribe_standard(audio, language, **kwargs)
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise
    
    def _transcribe_standard(
        self,
        audio: np.ndarray,
        language: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Transcribe using standard Whisper model."""
        # Config'den transkripsiyon parametrelerini al
        transcribe_params = self._get_transcribe_params(language, **kwargs)
        
        logger.debug(f"Transcribing audio - Language: {language or 'auto'}")
        
        # Transkripsiyon yap
        result = self.model.transcribe(
            audio,
            **transcribe_params
        )
        
        logger.info(f"Transcription completed - Detected language: {result.get('language', 'unknown')}")
        
        return result
    
    def _transcribe_transformers(
        self,
        audio: np.ndarray,
        language: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Transcribe using HuggingFace transformers model."""
        logger.debug(f"Transcribing audio with transformers - Language: {language or 'auto'}")
        
        # Ensure audio is float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        # Process audio
        inputs = self.processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt"
        )
        
        # Move to device and convert dtype to match model
        input_features = inputs.input_features.to(self.model.device)
        if hasattr(self.model, 'dtype'):
            input_features = input_features.to(self.model.dtype)
        
        # Set language token if specified
        forced_decoder_ids = None
        if language:
            forced_decoder_ids = self.processor.get_decoder_prompt_ids(
                language=language,
                task="transcribe"
            )
        
        # Generate transcription
        with torch.no_grad():
            predicted_ids = self.model.generate(
                input_features,
                forced_decoder_ids=forced_decoder_ids,
                max_length=448,
            )
        
        # Decode
        transcription = self.processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True
        )[0]
        
        logger.info(f"Transcription completed - Language: {language or 'auto'}")
        
        # Format output to match standard Whisper format
        return {
            'text': transcription,
            'language': language or 'unknown',
            'segments': [],  # Transformers don't provide segments by default
        }
    
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
        # Not: Whisper API'sine göre sadece desteklenen parametreleri kullan
        params = {
            'language': language or config.get('language.default'),
            'task': self.transcription_config.get('task', 'transcribe'),
            'temperature': self.transcription_config.get('temperature', 0.0),
            'beam_size': self.transcription_config.get('beam_size', 5),
            'best_of': self.transcription_config.get('best_of', 5),
            'patience': self.transcription_config.get('patience', 1.0),
            'length_penalty': self.transcription_config.get('length_penalty', 1.0),
            'compression_ratio_threshold': self.transcription_config.get('compression_ratio_threshold', 2.4),
            # 'log_prob_threshold': deprecated in newer Whisper versions
            'no_speech_threshold': self.transcription_config.get('no_speech_threshold', 0.6),
            'condition_on_previous_text': self.transcription_config.get('condition_on_previous_text', True),
            'initial_prompt': self.transcription_config.get('initial_prompt'),
            'word_timestamps': self.transcription_config.get('word_timestamps', True),
        }
        
        # Override parametreleri
        params.update(kwargs)
        
        return params
    
    def detect_language(self, audio: np.ndarray) -> str:
        """
        Ses verisinin dilini tespit et.
        
        Args:
            audio: Audio numpy array
        
        Returns:
            Detected language code
        """
        if not self._is_loaded:
            self.load()
        
        try:
            # Whisper'ın language detection özelliğini kullan
            audio_padded = whisper.pad_or_trim(audio)
            mel = whisper.log_mel_spectrogram(audio_padded).to(self.model.device)
            _, probs = self.model.detect_language(mel)
            
            detected_lang = max(probs, key=probs.get)
            confidence = probs[detected_lang]
            
            logger.debug(f"Language detected: {detected_lang} (confidence: {confidence:.2f})")
            return detected_lang
            
        except Exception as e:
            logger.error(f"Language detection failed: {e}")
            return 'en'  # fallback
    
    def format_output(
        self,
        result: Dict[str, Any],
        include_timestamps: bool = True,
        include_segments: bool = False,
    ) -> Dict[str, Any]:
        """
        Transkripsiyon çıktısını formatla.
        
        Args:
            result: Whisper transcription result
            include_timestamps: Timestamp bilgisi dahil et mi?
            include_segments: Segment detayları dahil et mi?
        
        Returns:
            Formatted output dictionary
        """
        output = {
            'text': result['text'].strip(),
            'language': result.get('language', 'unknown'),
        }
        
        if include_segments and 'segments' in result:
            segments = []
            for seg in result['segments']:
                segment_info = {
                    'text': seg['text'].strip(),
                }
                
                if include_timestamps:
                    segment_info.update({
                        'start': seg['start'],
                        'end': seg['end'],
                    })
                
                segments.append(segment_info)
            
            output['segments'] = segments
        
        return output
    
    def unload(self) -> None:
        """Model'i bellekten kaldır."""
        if self._is_loaded:
            del self.model
            self.model = None
            self._is_loaded = False
            
            # GPU memory temizliği
            if self.device in ['cuda', 'mps']:
                torch.cuda.empty_cache() if self.device == 'cuda' else None
            
            logger.info("Model unloaded from memory")
    
    @property
    def is_loaded(self) -> bool:
        """Model yüklü mü?"""
        return self._is_loaded
    
    def __repr__(self) -> str:
        return (f"WhisperASR(variant={self.variant}, device={self.device}, "
                f"loaded={self._is_loaded})")
    
    def __del__(self):
        """Destructor - model'i temizle."""
        self.unload()
