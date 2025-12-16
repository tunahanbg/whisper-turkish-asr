"""
Test quantized whisper-large-v3.w4a16 model.

This script tests the INT4 quantized model with the updated WhisperASR class.
"""

import sys
from pathlib import Path
import numpy as np
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.whisper_model import WhisperASR
import librosa


def test_quantized_model():
    """Test the quantized model with a sample audio file."""
    
    logger.info("="*70)
    logger.info("🧪 Testing INT4 Quantized Whisper Large v3")
    logger.info("="*70)
    
    # Model config
    model_config = {
        'name': 'whisper',
        'variant': 'large-v3',  # Not used, but kept for consistency
        'model_path': str(project_root / 'checkpoints' / 'hf_models' / 'whisper-large-v3-w4a16'),
        'device': 'cpu',  # Quantized model works best on CPU
        'compute_type': 'float16',
        'download_root': './checkpoints',
    }
    
    logger.info(f"📦 Model path: {model_config['model_path']}")
    logger.info(f"🖥️  Device: {model_config['device']}")
    
    # Initialize model
    try:
        logger.info("\n📥 Loading model...")
        asr = WhisperASR(custom_config=model_config)
        asr.load()
        logger.info("✅ Model loaded successfully!")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        return False
    
    # Test with a sample audio file
    test_audio_path = project_root / 'tests' / 'data' / 'test_set' / 'audio' / 'sample_001.flac'
    
    if not test_audio_path.exists():
        logger.warning(f"⚠️  Test audio not found: {test_audio_path}")
        logger.info("Creating dummy audio for basic test...")
        
        # Create 3 seconds of dummy audio (silence)
        audio = np.zeros(48000, dtype=np.float32)
        language = 'tr'
    else:
        logger.info(f"\n🎵 Loading test audio: {test_audio_path.name}")
        
        try:
            audio, sr = librosa.load(test_audio_path, sr=16000, mono=True)
            logger.info(f"   - Sample rate: {sr} Hz")
            logger.info(f"   - Duration: {len(audio)/sr:.2f}s")
            logger.info(f"   - Shape: {audio.shape}")
            language = 'tr'
        except Exception as e:
            logger.error(f"❌ Failed to load audio: {e}")
            return False
    
    # Transcribe
    try:
        logger.info("\n🎤 Transcribing...")
        result = asr.transcribe(audio, language=language)
        
        logger.info("="*70)
        logger.info("✅ TRANSCRIPTION RESULT")
        logger.info("="*70)
        logger.info(f"📝 Text: {result['text']}")
        logger.info(f"🌍 Language: {result['language']}")
        logger.info(f"📊 Segments: {len(result.get('segments', []))}")
        logger.info("="*70)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Transcription failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        asr.unload()
        logger.info("\n🧹 Model unloaded")


def compare_with_base():
    """Compare quantized large model with base model."""
    
    logger.info("\n" + "="*70)
    logger.info("⚖️  Comparing Quantized Large vs Base Model")
    logger.info("="*70)
    
    test_audio_path = project_root / 'tests' / 'data' / 'test_set' / 'audio' / 'sample_001.flac'
    
    if not test_audio_path.exists():
        logger.warning("⚠️  Test audio not found, skipping comparison")
        return
    
    # Load audio
    audio, sr = librosa.load(test_audio_path, sr=16000, mono=True)
    
    results = {}
    
    # Test quantized large
    logger.info("\n1️⃣  Testing Quantized Large v3...")
    try:
        import time
        
        asr_large = WhisperASR(custom_config={
            'name': 'whisper',
            'model_path': str(project_root / 'checkpoints' / 'hf_models' / 'whisper-large-v3-w4a16'),
            'device': 'cpu',
            'compute_type': 'float16',
        })
        asr_large.load()
        
        start = time.time()
        result_large = asr_large.transcribe(audio, language='tr')
        elapsed_large = time.time() - start
        
        results['large_quantized'] = {
            'text': result_large['text'],
            'time': elapsed_large,
        }
        
        logger.info(f"   ✅ Time: {elapsed_large:.2f}s")
        logger.info(f"   📝 Text: {result_large['text'][:100]}...")
        
        asr_large.unload()
        
    except Exception as e:
        logger.error(f"   ❌ Failed: {e}")
    
    # Test base model (faster-whisper)
    logger.info("\n2️⃣  Testing Base Model (faster-whisper)...")
    try:
        from src.models.faster_whisper_model import FasterWhisperASR
        import time
        
        asr_base = FasterWhisperASR(custom_config={
            'name': 'faster-whisper',
            'variant': 'base',
            'device': 'cpu',
            'compute_type': 'int8',
        })
        asr_base.load()
        
        start = time.time()
        result_base = asr_base.transcribe(audio, language='tr')
        elapsed_base = time.time() - start
        
        results['base_faster'] = {
            'text': result_base['text'],
            'time': elapsed_base,
        }
        
        logger.info(f"   ✅ Time: {elapsed_base:.2f}s")
        logger.info(f"   📝 Text: {result_base['text'][:100]}...")
        
        asr_base.unload()
        
    except Exception as e:
        logger.error(f"   ❌ Failed: {e}")
    
    # Summary
    if len(results) == 2:
        logger.info("\n" + "="*70)
        logger.info("📊 COMPARISON SUMMARY")
        logger.info("="*70)
        logger.info(f"⏱️  Speed:")
        logger.info(f"   - Quantized Large: {results['large_quantized']['time']:.2f}s")
        logger.info(f"   - Base (faster):   {results['base_faster']['time']:.2f}s")
        logger.info(f"   - Difference:      {abs(results['large_quantized']['time'] - results['base_faster']['time']):.2f}s")
        logger.info("="*70)


if __name__ == "__main__":
    success = test_quantized_model()
    
    if success:
        logger.info("\n✅ All tests passed!")
        
        # Optional: Compare with base model
        try:
            compare_with_base()
        except Exception as e:
            logger.warning(f"⚠️  Comparison skipped: {e}")
    else:
        logger.error("\n❌ Tests failed!")
        sys.exit(1)
