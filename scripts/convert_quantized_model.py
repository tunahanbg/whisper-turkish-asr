"""
Convert HuggingFace Whisper model to CTranslate2 format for faster-whisper.

This script downloads the quantized whisper-large-v3.w4a16 model from HuggingFace
and converts it to CTranslate2 format for use with faster-whisper.
"""

import os
import sys
from pathlib import Path
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def convert_model():
    """Convert HuggingFace model to CTranslate2 format."""
    try:
        from huggingface_hub import snapshot_download
        import ctranslate2
        logger.info("✅ Required packages available")
    except ImportError as e:
        logger.error(f"Missing package: {e}")
        logger.info("Installing required packages...")
        os.system("pip install -q huggingface_hub ctranslate2")
        from huggingface_hub import snapshot_download
        import ctranslate2
    
    # Paths
    hf_model_id = "nm-testing/whisper-large-v3.w4a16"
    download_dir = project_root / "checkpoints" / "hf_models" / "whisper-large-v3-w4a16"
    ct2_output_dir = project_root / "checkpoints" / "quantized_models" / "whisper-large-v3-w4a16-ct2"
    
    logger.info(f"🔽 Downloading model from HuggingFace: {hf_model_id}")
    logger.info(f"📂 Download directory: {download_dir}")
    
    # Create directories
    download_dir.parent.mkdir(parents=True, exist_ok=True)
    ct2_output_dir.parent.mkdir(parents=True, exist_ok=True)
    
    # Download model from HuggingFace
    try:
        logger.info("⬇️  Downloading model files...")
        model_path = snapshot_download(
            repo_id=hf_model_id,
            local_dir=str(download_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        logger.info(f"✅ Model downloaded to: {model_path}")
    except Exception as e:
        logger.error(f"❌ Download failed: {e}")
        raise
    
    # Convert to CTranslate2 format
    logger.info(f"🔄 Converting to CTranslate2 format...")
    logger.info(f"📂 Output directory: {ct2_output_dir}")
    
    try:
        # Use ct2-transformers-converter command
        convert_command = (
            f"ct2-transformers-converter "
            f"--model {download_dir} "
            f"--output_dir {ct2_output_dir} "
            f"--quantization int8_float16 "
            f"--force"
        )
        
        logger.info(f"Running: {convert_command}")
        result = os.system(convert_command)
        
        if result == 0:
            logger.info("✅ Conversion successful!")
            logger.info(f"📁 CT2 model location: {ct2_output_dir}")
            
            # Create a README
            readme_content = f"""# Whisper Large v3 (INT4 Quantized) - CTranslate2 Format

## Model Info
- **Source**: {hf_model_id}
- **Format**: CTranslate2 (for faster-whisper)
- **Quantization**: INT8_FLOAT16 (converted from INT4)
- **Converted**: {Path(__file__).name}

## Usage with faster-whisper

```python
from faster_whisper import WhisperModel

model = WhisperModel(
    "{ct2_output_dir}",
    device="cpu",
    compute_type="int8_float16"
)

segments, info = model.transcribe("audio.mp3", language="tr")
```

## Config.yaml Integration

```yaml
model:
    name: "faster-whisper"
    variant: "large-v3-w4a16"
    model_path: "{ct2_output_dir}"
    compute_type: "int8_float16"
```
"""
            with open(ct2_output_dir / "README.md", "w", encoding="utf-8") as f:
                f.write(readme_content)
            
            return str(ct2_output_dir)
        else:
            logger.error(f"❌ Conversion failed with exit code: {result}")
            return None
            
    except Exception as e:
        logger.error(f"❌ Conversion error: {e}")
        logger.info("💡 Trying alternative method...")
        
        # Alternative: Use Python API
        try:
            import transformers
            converter = ctranslate2.converters.TransformersConverter(str(download_dir))
            converter.convert(
                output_dir=str(ct2_output_dir),
                quantization="int8_float16",
                force=True
            )
            logger.info("✅ Alternative conversion successful!")
            return str(ct2_output_dir)
        except Exception as e2:
            logger.error(f"❌ Alternative method also failed: {e2}")
            return None


def verify_model(ct2_model_path: str):
    """Verify the converted model works."""
    logger.info("🧪 Testing converted model...")
    
    try:
        from faster_whisper import WhisperModel
        
        model = WhisperModel(
            ct2_model_path,
            device="cpu",
            compute_type="int8_float16"
        )
        logger.info("✅ Model loaded successfully!")
        logger.info(f"   - Device: cpu")
        logger.info(f"   - Compute type: int8_float16")
        
        # Test with dummy audio
        import numpy as np
        dummy_audio = np.zeros(16000, dtype=np.float32)  # 1 second of silence
        
        segments, info = model.transcribe(dummy_audio, language="tr")
        segments_list = list(segments)
        
        logger.info("✅ Model transcription test passed!")
        logger.info(f"   - Detected language: {info.language}")
        logger.info(f"   - Segments: {len(segments_list)}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Model verification failed: {e}")
        return False


if __name__ == "__main__":
    logger.info("="*70)
    logger.info("🚀 Converting Quantized Whisper Model to CTranslate2 Format")
    logger.info("="*70)
    
    # Convert model
    ct2_path = convert_model()
    
    if ct2_path:
        logger.info("="*70)
        logger.info("✅ CONVERSION COMPLETED")
        logger.info("="*70)
        logger.info(f"📂 Model location: {ct2_path}")
        logger.info("")
        logger.info("🔍 Verifying model...")
        
        if verify_model(ct2_path):
            logger.info("")
            logger.info("="*70)
            logger.info("✅ ALL TESTS PASSED!")
            logger.info("="*70)
            logger.info("")
            logger.info("📝 Next steps:")
            logger.info("1. Add to config.yaml:")
            logger.info(f'   model_path: "{ct2_path}"')
            logger.info('   compute_type: "int8_float16"')
            logger.info("")
            logger.info("2. Run benchmark test:")
            logger.info("   python tests/scripts/compare_models.py --models large-v3-w4a16 --limit 10")
            logger.info("")
        else:
            logger.warning("⚠️  Model verification failed, but conversion may still be OK")
    else:
        logger.error("="*70)
        logger.error("❌ CONVERSION FAILED")
        logger.error("="*70)
        sys.exit(1)
