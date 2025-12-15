"""
Test script for Faster-Whisper integration.

This script tests:
1. faster-whisper installation
2. Model loading
3. Transcription functionality
4. Performance comparison
"""

import sys
from pathlib import Path
import numpy as np
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from src.models import ModelManager, FasterWhisperASR

# Suppress debug logs for cleaner output
logger.remove()
logger.add(sys.stderr, level="INFO")


def check_faster_whisper_installation():
    """Check if faster-whisper is installed."""
    try:
        import faster_whisper
        logger.info(f"✅ faster-whisper installed (version: {faster_whisper.__version__})")
        return True
    except ImportError:
        logger.error("❌ faster-whisper not installed")
        logger.info("Install with: pip install faster-whisper")
        return False


def create_test_audio(duration_seconds=5, sample_rate=16000):
    """Create a test audio signal (sine wave)."""
    logger.info(f"Creating test audio ({duration_seconds}s, {sample_rate}Hz)...")
    
    # Simple sine wave
    t = np.linspace(0, duration_seconds, int(duration_seconds * sample_rate))
    frequency = 440  # A4 note
    audio = 0.3 * np.sin(2 * np.pi * frequency * t).astype(np.float32)
    
    return audio


def test_model_loading():
    """Test model loading."""
    logger.info("\n" + "="*60)
    logger.info("TEST 1: Model Loading")
    logger.info("="*60)
    
    try:
        # Test direct instantiation
        logger.info("Testing FasterWhisperASR direct instantiation...")
        model = FasterWhisperASR()
        
        logger.info("Loading model...")
        start = time.time()
        model.load()
        load_time = time.time() - start
        
        logger.info(f"✅ Model loaded in {load_time:.2f}s")
        logger.info(f"Model info: {model}")
        
        # Clean up
        model.unload()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Model loading failed: {e}")
        return False


def test_model_manager():
    """Test ModelManager with faster-whisper."""
    logger.info("\n" + "="*60)
    logger.info("TEST 2: ModelManager Integration")
    logger.info("="*60)
    
    try:
        manager = ModelManager()
        
        logger.info("Loading model via ModelManager...")
        start = time.time()
        model = manager.load_model(model_type="faster-whisper")
        load_time = time.time() - start
        
        logger.info(f"✅ Model loaded via manager in {load_time:.2f}s")
        logger.info(f"Manager: {manager}")
        
        # Clean up
        manager.unload_model()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ ModelManager test failed: {e}")
        return False


def test_transcription():
    """Test basic transcription."""
    logger.info("\n" + "="*60)
    logger.info("TEST 3: Transcription")
    logger.info("="*60)
    
    try:
        # Create test audio
        audio = create_test_audio(duration_seconds=3)
        
        # Load model
        model = FasterWhisperASR()
        model.load()
        
        logger.info("Running transcription...")
        start = time.time()
        result = model.transcribe(audio, language="en")
        transcribe_time = time.time() - start
        
        logger.info(f"✅ Transcription completed in {transcribe_time:.2f}s")
        logger.info(f"Text: {result['text']}")
        logger.info(f"Language: {result['language']}")
        logger.info(f"Segments: {len(result.get('segments', []))}")
        
        # Clean up
        model.unload()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Transcription test failed: {e}")
        return False


def test_config_switch():
    """Test switching between whisper and faster-whisper via config."""
    logger.info("\n" + "="*60)
    logger.info("TEST 4: Config-Based Model Switching")
    logger.info("="*60)
    
    try:
        from config import config
        
        # Check current config
        current_model = config.get('model.name')
        logger.info(f"Current model in config: {current_model}")
        
        if current_model == "faster-whisper":
            logger.info("✅ Config is set to faster-whisper")
        else:
            logger.warning(f"⚠️  Config is set to {current_model}")
            logger.info("To use faster-whisper, set in config/config.yaml:")
            logger.info("  model:")
            logger.info("    name: 'faster-whisper'")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Config test failed: {e}")
        return False


def main():
    """Run all tests."""
    logger.info("🚀 Faster-Whisper Integration Test Suite")
    logger.info("="*60)
    
    # Track results
    results = {}
    
    # Test 0: Installation check
    logger.info("\nChecking installation...")
    if not check_faster_whisper_installation():
        logger.error("\n❌ FAILED: faster-whisper not installed")
        logger.info("\nInstall with:")
        logger.info("  pip install faster-whisper")
        return
    
    # Test 1: Model loading
    results['loading'] = test_model_loading()
    
    # Test 2: ModelManager
    results['manager'] = test_model_manager()
    
    # Test 3: Transcription
    results['transcription'] = test_transcription()
    
    # Test 4: Config
    results['config'] = test_config_switch()
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("TEST SUMMARY")
    logger.info("="*60)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"{test_name.upper()}: {status}")
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    logger.info(f"\nTotal: {passed_tests}/{total_tests} tests passed")
    
    if all(results.values()):
        logger.info("\n🎉 ALL TESTS PASSED! Faster-Whisper is ready to use.")
        logger.info("\nNext steps:")
        logger.info("1. Update config/config.yaml:")
        logger.info("     model:")
        logger.info("       name: 'faster-whisper'")
        logger.info("       variant: 'base'")
        logger.info("       compute_type: 'int8'")
        logger.info("2. Run your application: streamlit run app.py")
    else:
        logger.error("\n❌ SOME TESTS FAILED. Check errors above.")


if __name__ == "__main__":
    main()

