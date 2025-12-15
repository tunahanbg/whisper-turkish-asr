"""
Audio preprocessing pipeline.
DRY: Tek bir class, config'den ayarlar.
"""

from typing import Optional
import numpy as np
import librosa
import noisereduce as nr
from loguru import logger

from config import config


class AudioPreprocessor:
    """
    Modüler ses ön işleme pipeline'ı.
    Her adım config'den kontrol edilebilir.
    """
    
    def __init__(self, custom_config: Optional[dict] = None):
        """
        Args:
            custom_config: Özel config (None = global config kullan)
        """
        self.config = custom_config or config.get('audio.preprocessing', {})
        self.sample_rate = config.get('audio.sample_rate', 16000)
        
        # Hangi adımların aktif olduğunu belirle
        self.enabled = self.config.get('enabled', True)
        self.normalize = self.config.get('normalize', True)
        self.trim_silence = self.config.get('trim_silence', True)
        self.trim_threshold_db = self.config.get('trim_threshold_db', 20)
        self.denoise = self.config.get('denoise', False)
        self.noise_reduction_strength = self.config.get('noise_reduction_strength', 0.8)
        
        logger.debug(f"AudioPreprocessor initialized - Normalize: {self.normalize}, "
                    f"Trim: {self.trim_silence}, Denoise: {self.denoise}")
    
    def process(
        self,
        audio: np.ndarray,
        sample_rate: Optional[int] = None,
    ) -> np.ndarray:
        """
        Ses verisini ön işle.
        
        Args:
            audio: Audio numpy array
            sample_rate: Sample rate (None = config'den al)
        
        Returns:
            Processed audio array
        """
        if not self.enabled:
            logger.debug("Preprocessing disabled, returning original audio")
            return audio
        
        sr = sample_rate or self.sample_rate
        
        try:
            # 1. Resample (gerekirse)
            if sr != self.sample_rate:
                audio = self._resample(audio, sr, self.sample_rate)
                sr = self.sample_rate
            
            # 2. Normalize
            if self.normalize:
                audio = self._normalize(audio)
            
            # 3. Trim silence
            if self.trim_silence:
                audio = self._trim_silence(audio)
            
            # 4. Denoise (opsiyonel)
            if self.denoise:
                audio = self._denoise(audio, sr)
            
            logger.debug(f"Audio preprocessed - Final shape: {audio.shape}")
            return audio
            
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            logger.warning("Returning original audio")
            return audio
    
    def _resample(
        self,
        audio: np.ndarray,
        orig_sr: int,
        target_sr: int,
    ) -> np.ndarray:
        """Resample audio to target sample rate."""
        logger.debug(f"Resampling: {orig_sr}Hz -> {target_sr}Hz")
        return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
    
    def _normalize(self, audio: np.ndarray) -> np.ndarray:
        """Peak normalization."""
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val
        logger.debug("Audio normalized")
        return audio
    
    def _trim_silence(self, audio: np.ndarray) -> np.ndarray:
        """Trim leading and trailing silence."""
        trimmed, _ = librosa.effects.trim(
            audio,
            top_db=self.trim_threshold_db,
        )
        logger.debug(f"Silence trimmed - Original: {len(audio)}, Trimmed: {len(trimmed)}")
        return trimmed
    
    def _denoise(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Reduce noise using spectral gating."""
        logger.debug("Applying noise reduction")
        denoised = nr.reduce_noise(
            y=audio,
            sr=sr,
            prop_decrease=self.noise_reduction_strength,
            stationary=True,
        )
        return denoised
    
    def __repr__(self) -> str:
        return (f"AudioPreprocessor(enabled={self.enabled}, "
                f"normalize={self.normalize}, trim={self.trim_silence}, "
                f"denoise={self.denoise})")

