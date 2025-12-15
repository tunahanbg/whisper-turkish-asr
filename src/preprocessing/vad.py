"""
Voice Activity Detection using Silero VAD.
DRY: Config-driven VAD modülü.
"""

from typing import List, Tuple, Optional
import numpy as np
import torch
from loguru import logger

from config import config


class VoiceActivityDetector:
    """
    Silero VAD kullanarak konuşma tespiti.
    Config dosyasından tüm parametreleri alır.
    """
    
    def __init__(self, custom_config: Optional[dict] = None):
        """
        Args:
            custom_config: Özel VAD config (None = global config)
        """
        self.config = custom_config or config.get('vad', {})
        self.enabled = self.config.get('enabled', True)
        
        if not self.enabled:
            logger.info("VAD disabled")
            return
        
        # VAD parametreleri
        self.threshold = self.config.get('threshold', 0.5)
        self.min_speech_duration_ms = self.config.get('min_speech_duration_ms', 250)
        self.max_speech_duration_s = self.config.get('max_speech_duration_s', 600)
        self.min_silence_duration_ms = self.config.get('min_silence_duration_ms', 10000)
        self.window_size_samples = self.config.get('window_size_samples', 512)
        self.speech_pad_ms = self.config.get('speech_pad_ms', 30)
        
        # Sample rate
        self.sample_rate = config.get('audio.sample_rate', 16000)
        
        # Model'i yükle
        self.model = None
        self.utils = None
        self._load_model()
        
        logger.info(f"VAD initialized - Threshold: {self.threshold}, "
                   f"Min silence: {self.min_silence_duration_ms}ms")
    
    def _load_model(self) -> None:
        """Silero VAD modelini yükle."""
        try:
            model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False,
            )
            
            self.model = model
            self.utils = utils
            
            logger.info("Silero VAD model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load VAD model: {e}")
            self.enabled = False
            raise
    
    def detect_speech(
        self,
        audio: np.ndarray,
        sample_rate: Optional[int] = None,
    ) -> List[dict]:
        """
        Ses verisinde konuşma segmentlerini tespit et.
        
        Args:
            audio: Audio numpy array
            sample_rate: Sample rate (None = config'den al)
        
        Returns:
            List of speech segments with start/end timestamps
            [{'start': 0.5, 'end': 3.2}, ...]
        """
        if not self.enabled:
            logger.warning("VAD disabled, returning full audio as single segment")
            duration = len(audio) / (sample_rate or self.sample_rate)
            return [{'start': 0.0, 'end': duration}]
        
        sr = sample_rate or self.sample_rate
        
        try:
            # Audio'yu torch tensor'a çevir
            wav = torch.from_numpy(audio).float()
            
            # Speech timestamps al
            get_speech_timestamps = self.utils[0]
            
            speech_timestamps = get_speech_timestamps(
                wav,
                self.model,
                sampling_rate=sr,
                threshold=self.threshold,
                min_speech_duration_ms=self.min_speech_duration_ms,
                max_speech_duration_s=self.max_speech_duration_s,
                min_silence_duration_ms=self.min_silence_duration_ms,
                window_size_samples=self.window_size_samples,
                speech_pad_ms=self.speech_pad_ms,
            )
            
            # Sample index'leri saniyeye çevir
            segments = []
            for ts in speech_timestamps:
                segment = {
                    'start': ts['start'] / sr,
                    'end': ts['end'] / sr,
                }
                segments.append(segment)
            
            logger.debug(f"Detected {len(segments)} speech segments")
            return segments
            
        except Exception as e:
            logger.error(f"Speech detection failed: {e}")
            return []
    
    def has_speech(
        self,
        audio: np.ndarray,
        sample_rate: Optional[int] = None,
    ) -> bool:
        """
        Ses verisinde konuşma var mı kontrol et.
        
        Args:
            audio: Audio numpy array
            sample_rate: Sample rate
        
        Returns:
            True if speech detected, False otherwise
        """
        segments = self.detect_speech(audio, sample_rate)
        return len(segments) > 0
    
    def check_silence_duration(
        self,
        audio: np.ndarray,
        sample_rate: Optional[int] = None,
    ) -> float:
        """
        Sondaki sessizlik süresini hesapla.
        
        Args:
            audio: Audio numpy array
            sample_rate: Sample rate
        
        Returns:
            Trailing silence duration (saniye)
        """
        sr = sample_rate or self.sample_rate
        segments = self.detect_speech(audio, sr)
        
        if not segments:
            return len(audio) / sr
        
        last_speech_end = segments[-1]['end']
        total_duration = len(audio) / sr
        
        silence_duration = total_duration - last_speech_end
        return silence_duration
    
    def should_stop_recording(
        self,
        audio: np.ndarray,
        sample_rate: Optional[int] = None,
    ) -> bool:
        """
        Kayıt durdurulmalı mı? (Yeterince sessizlik var mı?)
        
        Args:
            audio: Audio numpy array
            sample_rate: Sample rate
        
        Returns:
            True if recording should stop
        """
        silence_duration = self.check_silence_duration(audio, sample_rate)
        threshold_seconds = self.min_silence_duration_ms / 1000.0
        
        should_stop = silence_duration >= threshold_seconds
        
        if should_stop:
            logger.info(f"Silence threshold reached: {silence_duration:.2f}s >= {threshold_seconds:.2f}s")
        
        return should_stop
    
    def __repr__(self) -> str:
        return (f"VoiceActivityDetector(enabled={self.enabled}, "
                f"threshold={self.threshold})")

