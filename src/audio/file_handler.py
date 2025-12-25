"""
Audio file handling utilities.
Ses dosyası yükleme, kaydetme, format dönüşümü.
"""

from pathlib import Path
from typing import Union, Optional, Tuple
import numpy as np
from pydub import AudioSegment
from loguru import logger

from config import config
from src.utils.audio_utils import load_audio, save_audio, validate_audio


class AudioFileHandler:
    """
    Ses dosyası işlemleri için utility class.
    Farklı formatları destekler, config'den ayarları alır.
    """
    
    def __init__(self):
        """Initialize file handler."""
        self.sample_rate = config.get('audio.sample_rate', 16000)
        self.format = config.get('audio.format', 'wav')
        self.supported_formats = config.get('ui.supported_formats', 
                                           ['wav', 'mp3', 'm4a', 'ogg', 'flac'])
        
        logger.debug(f"AudioFileHandler initialized - Default format: {self.format}")
    
    def load(
        self,
        file_path: Union[str, Path],
        **kwargs
    ) -> Tuple[np.ndarray, int]:
        """
        Ses dosyasını yükle.
        
        Args:
            file_path: Dosya yolu
            **kwargs: load_audio için ek parametreler
        
        Returns:
            (audio, sample_rate) tuple
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        # Format kontrolü
        if file_path.suffix[1:] not in self.supported_formats:
            logger.warning(f"File format {file_path.suffix} may not be supported")
        
        try:
            audio, sr = load_audio(file_path, sample_rate=self.sample_rate, **kwargs)
            
            # Validation
            if not validate_audio(audio, sr):
                raise ValueError("Audio validation failed")
            
            return audio, sr
            
        except Exception as e:
            logger.error(f"Failed to load audio file: {e}")
            raise
    
    def save(
        self,
        audio: np.ndarray,
        file_path: Union[str, Path],
        sample_rate: Optional[int] = None,
        format: Optional[str] = None,
    ) -> None:
        """
        Ses verisini dosyaya kaydet.
        
        Args:
            audio: Audio numpy array
            file_path: Hedef dosya yolu
            sample_rate: Sample rate (None = config'den al)
            format: Dosya formatı (None = config'den al)
        """
        sr = sample_rate or self.sample_rate
        fmt = format or self.format
        
        try:
            save_audio(audio, file_path, sr, format=fmt)
            logger.info(f"Audio saved: {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to save audio: {e}")
            raise
    
    def convert_format(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        output_format: str = 'wav',
    ) -> None:
        """
        Ses dosyası formatını dönüştür.
        
        Args:
            input_path: Girdi dosyası
            output_path: Çıktı dosyası
            output_format: Hedef format
        """
        try:
            logger.info(f"Converting {input_path} to {output_format}")
            
            # Pydub ile yükle
            audio = AudioSegment.from_file(input_path)
            
            # Export
            audio.export(
                output_path,
                format=output_format,
                parameters=["-ar", str(self.sample_rate)]
            )
            
            logger.info(f"Conversion completed: {output_path}")
            
        except Exception as e:
            logger.error(f"Format conversion failed: {e}")
            raise
    
    def get_file_info(self, file_path: Union[str, Path]) -> dict:
        """
        Ses dosyası hakkında bilgi al.
        
        Args:
            file_path: Dosya yolu
        
        Returns:
            File info dictionary
        """
        try:
            file_path = Path(file_path)
            audio, sr = load_audio(file_path)
            
            duration = len(audio) / sr
            size_mb = file_path.stat().st_size / (1024 * 1024)
            
            info = {
                'path': str(file_path),
                'format': file_path.suffix[1:],
                'duration': duration,
                'sample_rate': sr,
                'samples': len(audio),
                'channels': 1,  # mono
                'size_mb': size_mb,
            }
            
            return info
            
        except Exception as e:
            logger.error(f"Failed to get file info: {e}")
            raise
    
    def validate_file(
        self,
        file_path: Union[str, Path],
        max_size_mb: Optional[float] = None,
    ) -> bool:
        """
        Dosya geçerli mi kontrol et.
        
        Args:
            file_path: Dosya yolu
            max_size_mb: Maximum dosya boyutu (MB)
        
        Returns:
            True if valid, False otherwise
        """
        try:
            file_path = Path(file_path)
            
            # Dosya var mı?
            if not file_path.exists():
                logger.warning(f"File does not exist: {file_path}")
                return False
            
            # Format destekleniyor mu?
            if file_path.suffix[1:] not in self.supported_formats:
                logger.warning(f"Unsupported format: {file_path.suffix}")
                return False
            
            # Boyut kontrolü
            if max_size_mb:
                size_mb = file_path.stat().st_size / (1024 * 1024)
                if size_mb > max_size_mb:
                    logger.warning(f"File too large: {size_mb:.1f}MB > {max_size_mb}MB")
                    return False
            
            # Audio yükleyip validate et
            audio, sr = load_audio(file_path)
            return validate_audio(audio, sr)
            
        except Exception as e:
            logger.error(f"File validation failed: {e}")
            return False
    
    def __repr__(self) -> str:
        return f"AudioFileHandler(format={self.format}, sr={self.sample_rate})"

