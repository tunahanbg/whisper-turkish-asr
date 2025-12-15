"""
Audio utility functions.
DRY: Merkezi ses işleme utility fonksiyonları.
"""

from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np
import librosa
import soundfile as sf
from loguru import logger

from config import config


def load_audio(
    file_path: Union[str, Path],
    sample_rate: Optional[int] = None,
    mono: bool = True,
    offset: float = 0.0,
    duration: Optional[float] = None,
) -> Tuple[np.ndarray, int]:
    """
    Ses dosyasını yükle.
    
    Args:
        file_path: Ses dosyası yolu
        sample_rate: Hedef sample rate (None = orijinal)
        mono: Mono'ya çevir mi?
        offset: Başlangıç zamanı (saniye)
        duration: Yüklenecek süre (saniye)
    
    Returns:
        (audio_data, sample_rate) tuple
    """
    try:
        # Config'den default sample rate al
        if sample_rate is None:
            sample_rate = config.get('audio.sample_rate', 16000)
        
        audio, sr = librosa.load(
            file_path,
            sr=sample_rate,
            mono=mono,
            offset=offset,
            duration=duration,
        )
        
        logger.debug(f"Audio loaded: {file_path} - Shape: {audio.shape}, SR: {sr}")
        return audio, sr
        
    except Exception as e:
        logger.error(f"Failed to load audio {file_path}: {e}")
        raise


def save_audio(
    audio: np.ndarray,
    file_path: Union[str, Path],
    sample_rate: int,
    format: Optional[str] = None,
) -> None:
    """
    Ses verisini dosyaya kaydet.
    
    Args:
        audio: Audio numpy array
        file_path: Hedef dosya yolu
        sample_rate: Sample rate
        format: Dosya formatı (None = uzantıdan belirle)
    """
    try:
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        sf.write(file_path, audio, sample_rate, format=format)
        logger.debug(f"Audio saved: {file_path}")
        
    except Exception as e:
        logger.error(f"Failed to save audio {file_path}: {e}")
        raise


def validate_audio(
    audio: np.ndarray,
    sample_rate: int,
    min_duration: float = 0.1,
    max_duration: Optional[float] = None,
) -> bool:
    """
    Ses verisinin geçerliliğini kontrol et.
    
    Args:
        audio: Audio numpy array
        sample_rate: Sample rate
        min_duration: Minimum süre (saniye)
        max_duration: Maximum süre (saniye, None = sınırsız)
    
    Returns:
        True if valid, False otherwise
    """
    try:
        duration = len(audio) / sample_rate
        
        if duration < min_duration:
            logger.warning(f"Audio too short: {duration:.2f}s < {min_duration}s")
            return False
        
        if max_duration and duration > max_duration:
            logger.warning(f"Audio too long: {duration:.2f}s > {max_duration}s")
            return False
        
        if np.all(audio == 0):
            logger.warning("Audio contains only silence")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Audio validation failed: {e}")
        return False


def get_audio_duration(
    audio: Union[np.ndarray, str, Path],
    sample_rate: Optional[int] = None,
) -> float:
    """
    Ses süresini hesapla.
    
    Args:
        audio: Audio array veya dosya yolu
        sample_rate: Sample rate (array için gerekli)
    
    Returns:
        Süre (saniye)
    """
    try:
        if isinstance(audio, (str, Path)):
            audio, sample_rate = load_audio(audio, sample_rate=sample_rate)
        
        if sample_rate is None:
            raise ValueError("sample_rate is required for numpy array")
        
        duration = len(audio) / sample_rate
        return duration
        
    except Exception as e:
        logger.error(f"Failed to get audio duration: {e}")
        raise


def format_timestamp(seconds: float) -> str:
    """
    Saniyeyi HH:MM:SS formatına çevir.
    
    Args:
        seconds: Saniye cinsinden zaman
    
    Returns:
        Formatlanmış zaman string'i
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
    else:
        return f"{minutes:02d}:{secs:02d}.{millis:03d}"

