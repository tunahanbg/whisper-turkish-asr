"""
Audio recorder with VAD support.
DRY: Config-driven mikrofon kaydı, VAD entegrasyonu.
"""

from typing import Optional, Callable
import threading
import queue
import numpy as np
import sounddevice as sd
from loguru import logger

from config import config
from src.preprocessing.vad import VoiceActivityDetector


class AudioRecorder:
    """
    Real-time mikrofon kaydı.
    VAD ile otomatik durdurma desteği.
    """
    
    def __init__(
        self,
        vad: Optional[VoiceActivityDetector] = None,
        callback: Optional[Callable] = None,
    ):
        """
        Args:
            vad: VoiceActivityDetector instance (None = yeni oluştur)
            callback: Her chunk için callback fonksiyonu
        """
        # Config'den ayarlar
        self.recording_config = config.get('recording', {})
        self.sample_rate = config.get('audio.sample_rate', 16000)
        self.channels = config.get('audio.channels', 1)
        
        # Recording parametreleri
        self.device = self.recording_config.get('device')
        self.auto_stop = self.recording_config.get('auto_stop_silence', True)
        self.max_duration = self.recording_config.get('max_recording_duration', 600)
        self.buffer_size = self.recording_config.get('buffer_size', 1024)
        
        # VAD
        self.vad = vad or VoiceActivityDetector()
        
        # Callback
        self.callback = callback
        
        # Recording state
        self.is_recording = False
        self.audio_queue = queue.Queue()
        self.recorded_audio = []
        self.stream = None
        
        logger.debug(f"AudioRecorder initialized - SR: {self.sample_rate}Hz, "
                    f"Auto-stop: {self.auto_stop}")
    
    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info,
        status
    ) -> None:
        """Sounddevice callback - her chunk için çağrılır."""
        if status:
            logger.warning(f"Recording status: {status}")
        
        # Mono'ya çevir
        audio_chunk = indata[:, 0].copy() if indata.ndim > 1 else indata.copy()
        
        # Queue'ya ekle
        self.audio_queue.put(audio_chunk)
        
        # Custom callback varsa çağır
        if self.callback:
            self.callback(audio_chunk)
    
    def start_recording(self) -> None:
        """Kaydı başlat."""
        if self.is_recording:
            logger.warning("Recording already in progress")
            return
        
        try:
            # State'i sıfırla
            self.is_recording = True
            self.recorded_audio = []
            self.audio_queue = queue.Queue()
            
            # Stream'i aç
            self.stream = sd.InputStream(
                device=self.device,
                channels=self.channels,
                samplerate=self.sample_rate,
                blocksize=self.buffer_size,
                callback=self._audio_callback,
            )
            
            self.stream.start()
            logger.info("Recording started")
            
            # Processing thread'i başlat
            self.processing_thread = threading.Thread(
                target=self._process_audio,
                daemon=True,
            )
            self.processing_thread.start()
            
        except Exception as e:
            logger.error(f"Failed to start recording: {e}")
            self.is_recording = False
            raise
    
    def _process_audio(self) -> None:
        """Audio chunk'ları işle (background thread)."""
        while self.is_recording:
            try:
                # Queue'dan chunk al
                chunk = self.audio_queue.get(timeout=0.1)
                self.recorded_audio.append(chunk)
                
                # Auto-stop kontrolü (VAD)
                if self.auto_stop and self.vad.enabled:
                    full_audio = np.concatenate(self.recorded_audio)
                    
                    # Maximum süre kontrolü
                    duration = len(full_audio) / self.sample_rate
                    if duration >= self.max_duration:
                        logger.info(f"Max duration reached: {duration:.1f}s")
                        self.stop_recording()
                        break
                    
                    # Sessizlik kontrolü
                    if self.vad.should_stop_recording(full_audio, self.sample_rate):
                        logger.info("Silence detected, stopping recording")
                        self.stop_recording()
                        break
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing audio: {e}")
                break
    
    def stop_recording(self) -> np.ndarray:
        """
        Kaydı durdur ve audio'yu döndür.
        
        Returns:
            Recorded audio as numpy array
        """
        if not self.is_recording:
            logger.warning("No recording in progress")
            return np.array([])
        
        try:
            # Recording'i durdur
            self.is_recording = False
            
            # Stream'i kapat
            if self.stream:
                self.stream.stop()
                self.stream.close()
                self.stream = None
            
            # Thread'in bitmesini bekle
            if hasattr(self, 'processing_thread'):
                self.processing_thread.join(timeout=1.0)
            
            # Queue'daki kalan chunk'ları al
            while not self.audio_queue.empty():
                try:
                    chunk = self.audio_queue.get_nowait()
                    self.recorded_audio.append(chunk)
                except queue.Empty:
                    break
            
            # Audio'yu birleştir
            if self.recorded_audio:
                audio = np.concatenate(self.recorded_audio)
                duration = len(audio) / self.sample_rate
                logger.info(f"Recording stopped - Duration: {duration:.2f}s, "
                          f"Samples: {len(audio)}")
                return audio
            else:
                logger.warning("No audio recorded")
                return np.array([])
            
        except Exception as e:
            logger.error(f"Error stopping recording: {e}")
            return np.array([])
    
    def get_recording_duration(self) -> float:
        """Mevcut kayıt süresini döndür (saniye)."""
        if not self.recorded_audio:
            return 0.0
        
        total_samples = sum(len(chunk) for chunk in self.recorded_audio)
        duration = total_samples / self.sample_rate
        return duration
    
    @staticmethod
    def list_devices() -> None:
        """Mevcut ses cihazlarını listele."""
        print("\nAvailable Audio Devices:")
        print(sd.query_devices())
    
    def __repr__(self) -> str:
        return (f"AudioRecorder(sr={self.sample_rate}, "
                f"recording={self.is_recording}, "
                f"auto_stop={self.auto_stop})")

