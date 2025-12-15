"""
Basic usage examples for ASR system.
DRY: Modüler kod kullanım örnekleri.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from loguru import logger

from config import config
from src.models import ModelManager
from src.preprocessing import AudioPreprocessor, VoiceActivityDetector
from src.audio import AudioFileHandler
from src.utils.audio_utils import format_timestamp


def example_transcribe_file(file_path: str):
    """
    Örnek: Ses dosyasını transkribe et.
    
    Args:
        file_path: Ses dosyası yolu
    """
    logger.info(f"Transcribing file: {file_path}")
    
    # 1. Model Manager'ı başlat
    model_manager = ModelManager()
    model_manager.load_model()
    model = model_manager.get_model()
    
    # 2. Ses dosyasını yükle
    file_handler = AudioFileHandler()
    audio, sr = file_handler.load(file_path)
    
    logger.info(f"Audio loaded - Duration: {len(audio)/sr:.2f}s")
    
    # 3. Ön işleme
    preprocessor = AudioPreprocessor()
    processed_audio = preprocessor.process(audio, sr)
    
    # 4. Transkripsiyon
    result = model.transcribe(processed_audio, language='tr')
    
    # 5. Sonucu göster
    formatted_result = model.format_output(result, include_timestamps=True, include_segments=True)
    
    print("\n" + "="*80)
    print("TRANSKRIPSIYON SONUCU")
    print("="*80)
    print(f"\nMetin: {formatted_result['text']}")
    print(f"Dil: {formatted_result['language']}")
    
    if 'segments' in formatted_result:
        print("\nSegmentler:")
        for i, seg in enumerate(formatted_result['segments'], 1):
            start = format_timestamp(seg['start'])
            end = format_timestamp(seg['end'])
            print(f"  [{start} → {end}] {seg['text']}")
    
    print("="*80)
    
    # Cleanup
    model_manager.unload_model()


def example_vad_detection():
    """Örnek: VAD ile konuşma tespiti."""
    logger.info("VAD detection example")
    
    # Dummy audio oluştur (gerçek kullanımda dosyadan yüklenecek)
    sr = 16000
    duration = 5  # saniye
    
    # Konuşma simülasyonu: 1-3 saniye arası sinyal var
    audio = np.zeros(sr * duration)
    audio[sr * 1:sr * 3] = np.random.randn(sr * 2) * 0.5
    
    # VAD detector
    vad = VoiceActivityDetector()
    segments = vad.detect_speech(audio, sr)
    
    print("\n" + "="*80)
    print("VAD DETECTION SONUCU")
    print("="*80)
    print(f"Tespit edilen segment sayısı: {len(segments)}")
    
    for i, seg in enumerate(segments, 1):
        start = format_timestamp(seg['start'])
        end = format_timestamp(seg['end'])
        duration = seg['end'] - seg['start']
        print(f"Segment {i}: [{start} → {end}] ({duration:.2f}s)")
    
    print("="*80)


def example_config_usage():
    """Örnek: Config kullanımı."""
    print("\n" + "="*80)
    print("CONFIG KULLANIMI")
    print("="*80)
    
    # Model ayarları
    print(f"Model: {config.get('model.name')} {config.get('model.variant')}")
    print(f"Device: {config.get('model.device')}")
    
    # Audio ayarları
    print(f"Sample Rate: {config.get('audio.sample_rate')}Hz")
    print(f"Preprocessing Enabled: {config.get('audio.preprocessing.enabled')}")
    
    # VAD ayarları
    print(f"VAD Enabled: {config.get('vad.enabled')}")
    print(f"VAD Threshold: {config.get('vad.threshold')}")
    
    # Runtime'da config güncelleme
    config.set('model.variant', 'small')
    print(f"\nGüncellenen Model Variant: {config.get('model.variant')}")
    
    # Geri al
    config.set('model.variant', 'medium')
    
    print("="*80)


def main():
    """Ana fonksiyon."""
    print("\n🎤 ASR System - Usage Examples\n")
    
    # Config örneği
    example_config_usage()
    
    # VAD örneği
    example_vad_detection()
    
    # Dosya transkripsiyon örneği
    # Not: Gerçek ses dosyası gerekli
    # example_transcribe_file("path/to/audio.wav")
    
    print("\n✅ Examples completed!\n")


if __name__ == "__main__":
    main()

