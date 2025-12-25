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
        input_gain: float = 3.0,  # Mikrofon gain (düşük sinyali artır)
    ):
        """
        Args:
            vad: VoiceActivityDetector instance (None = yeni oluştur)
            callback: Her chunk için callback fonksiyonu
            input_gain: Mikrofon sinyalini artırmak için çarpan (default: 3.0)
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
        self.input_gain = input_gain  # Mikrofon gain çarpanı
        
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
                    f"Auto-stop: {self.auto_stop}, Input gain: {self.input_gain}x")
    
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
        
        # Mono'ya çevir (channels > 1 ise)
        audio_chunk = indata[:, 0].copy() if indata.ndim > 1 else indata.copy()
        
        # DEBUG: İlk chunk'ta dtype ve range kontrolü (BEFORE gain)
        is_first_chunk = len(self.recorded_audio) == 0
        if is_first_chunk:
            logger.debug(f"First audio chunk (before gain) - dtype: {audio_chunk.dtype}, "
                        f"shape: {audio_chunk.shape}, "
                        f"range: [{audio_chunk.min():.4f}, {audio_chunk.max():.4f}]")
        
        # Input gain uygula (mikrofon sinyali düşükse amplify et)
        if self.input_gain != 1.0:
            audio_chunk = audio_chunk * self.input_gain
            # Clipping kontrolü [-1, 1] aralığında kal
            audio_chunk = np.clip(audio_chunk, -1.0, 1.0)
            
            # İlk chunk'ta gain uygulandığını logla
            if is_first_chunk:
                logger.info(f"Input gain applied: {self.input_gain}x (after: [{audio_chunk.min():.4f}, {audio_chunk.max():.4f}])")
        
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
                dtype='float32',  # Whisper float32 bekliyor
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
                        # Thread içinden stop_recording çağırma, sadece flag'i değiştir
                        self.is_recording = False
                        break
                    
                    # Sessizlik kontrolü
                    if self.vad.should_stop_recording(full_audio, self.sample_rate):
                        logger.info("Silence detected, auto-stopping")
                        # Thread içinden stop_recording çağırma, sadece flag'i değiştir
                        self.is_recording = False
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
        # Double-stop koruması
        if not self.is_recording and not self.recorded_audio and not self.stream:
            logger.warning("No recording in progress")
            return np.array([])
        
        try:
            # Recording'i durdur (eğer hala çalışıyorsa)
            was_recording = self.is_recording
            self.is_recording = False
            
            # Stream'i HER DURUMDA kapat 
            if self.stream:
                try:
                    logger.debug("Closing audio stream...")
                    self.stream.stop()
                    self.stream.close()
                    logger.debug("Audio stream closed successfully")
                except Exception as e:
                    logger.warning(f"Error closing stream: {e}")
                finally:
                    self.stream = None
            
            # Thread'in bitmesini bekle (timeout ile, segfault önleme)
            if hasattr(self, 'processing_thread') and self.processing_thread:
                try:
                    logger.debug("Waiting for processing thread to finish...")
                    self.processing_thread.join(timeout=2.0)
                    if self.processing_thread.is_alive():
                        logger.warning("Processing thread did not finish in time")
                    else:
                        logger.debug("Processing thread finished successfully")
                except Exception as e:
                    logger.warning(f"Error joining thread: {e}")
                finally:
                    self.processing_thread = None
            
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
                
                # DEBUG: Audio detaylarını logla
                logger.info(f"Recording stopped - Duration: {duration:.2f}s, "
                          f"Samples: {len(audio)}, dtype: {audio.dtype}, "
                          f"shape: {audio.shape}")
                logger.debug(f"Audio stats - min: {audio.min():.4f}, "
                            f"max: {audio.max():.4f}, "
                            f"mean: {audio.mean():.4f}, "
                            f"std: {audio.std():.4f}")
                
                # Audio sessiz mi kontrolü
                if np.abs(audio).max() < 0.001:
                    logger.warning("Audio is nearly silent! Max amplitude < 0.001")
                
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
                f"auto_stop={self.auto_stop}, "
                f"gain={self.input_gain}x)")

