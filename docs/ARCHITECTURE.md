# ASR System - Architecture & Module Flow

## 📐 System Architecture Overview

This document provides a deep technical overview of how the ASR system works, module dependencies, and execution flow.

---

## 🎯 Core Design Principles

1. **Config-Driven**: All parameters centralized in `config/config.yaml`
2. **Modular**: Each module has single responsibility
3. **Factory Pattern**: Easy model switching via ModelManager
4. **Singleton Pattern**: Single Config instance across application
5. **DRY**: No code duplication, reusable utilities

---

## 🔄 Execution Flow

### 1. Application Startup (Streamlit UI)

```
app.py (entry point)
  └─> src/ui/app.py::main()
      ├─> initialize_session_state()
      │   ├─> ModelManager (lazy, not loaded yet)
      │   ├─> AudioPreprocessor
      │   ├─> VoiceActivityDetector
      │   └─> AudioFileHandler
      └─> sidebar_settings() + main tabs
```

### 2. Model Loading Flow

```
User clicks "Load Model" or starts transcription
  └─> load_model() in ui/app.py
      └─> ModelManager.load_model()
          ├─> Read config (model.name, model.variant, model.device)
          ├─> Factory pattern selects implementation:
          │   ├─> faster-whisper → FasterWhisperASR
          │   └─> whisper → WhisperASR
          └─> model.load()
              ├─> FasterWhisperASR: Load CTranslate2 model
              └─> WhisperASR: Load OpenAI Whisper or HF Transformers (quantized)
```

**Model Types:**
- **Faster-Whisper** (Primary): CTranslate2 optimized, 3-4x faster, int8 quantization
- **Whisper (Standard)**: OpenAI's original implementation
- **Whisper (Quantized)**: HuggingFace Transformers with INT4 quantization

### 3. File Upload Transcription Flow

```
User uploads audio file
  └─> file_upload_tab()
      └─> AudioFileHandler.load(file_path)
          ├─> librosa.load() → numpy array (mono, 16kHz)
          └─> validate_audio()
      └─> transcribe_audio(audio, language)
          ├─> load_model() (if not loaded)
          ├─> Normalize audio ([-1, 1] range)
          ├─> Check silence (RMS < 0.01 → reject)
          └─> model.transcribe(audio, language)
              ├─> FasterWhisperASR.transcribe()
              │   ├─> WhisperModel.transcribe() (CTranslate2)
              │   └─> Return segments + text
              └─> WhisperASR.transcribe()
                  ├─> Standard: whisper.transcribe()
                  └─> Quantized: HF generate() + decode()
          └─> format_output() → display_transcription_result()
```

### 4. Microphone Recording Flow

```
User clicks "Start Recording"
  └─> microphone_tab()
      └─> AudioRecorder(vad, input_gain)
          └─> start_recording()
              ├─> sounddevice.InputStream (float32, 16kHz, mono)
              ├─> Background thread: _process_audio()
              │   ├─> Collect chunks in queue
              │   ├─> Apply input_gain (amplify signal)
              │   └─> VAD check: should_stop_recording()
              │       └─> VoiceActivityDetector.should_stop_recording()
              │           ├─> Silero VAD model
              │           ├─> detect_speech() → segments
              │           └─> check_silence_duration()
              └─> Auto-stop when silence > 10s

User clicks "Stop Recording" (or VAD auto-stops)
  └─> stop_recording()
      ├─> Close stream
      ├─> Concatenate chunks → numpy array
      └─> transcribe_audio(audio, language)
          └─> [Same as file upload flow]
```

---

## 📦 Module Dependencies

### Core Modules

#### 1. Config System (`config/`)
```
config/__init__.py (Config class - Singleton)
  └─> config.yaml (YAML file)
      ├─> model: name, variant, device, compute_type
      ├─> audio: sample_rate, preprocessing settings
      ├─> vad: threshold, silence duration
      ├─> transcription: beam_size, temperature, etc.
      └─> evaluation: test settings
```

**Key Methods:**
- `config.get('model.variant')` - Nested key access
- `config.set('model.variant', 'small')` - Runtime updates
- `config.model_config` - Property shortcuts

#### 2. Model Management (`src/models/`)

**Hierarchy:**
```
BaseASR (abstract)
  ├─> WhisperASR (Standard + Quantized HF)
  └─> FasterWhisperASR (CTranslate2)

ModelManager (Factory)
  └─> Selects implementation based on config
```

**FasterWhisperASR** (Primary):
- Uses CTranslate2 backend
- 3-4x faster than standard Whisper
- int8 quantization by default
- Supports custom/fine-tuned models
- Device: CPU, CUDA (no MPS support)

**WhisperASR**:
- Standard: OpenAI whisper library
- Quantized: HuggingFace Transformers
  - INT4 quantization (compressed-tensors)
  - Large-v3 model: WER 19% (best accuracy)
  - Slower but more accurate

#### 3. Audio Processing (`src/audio/`)

**AudioFileHandler**:
- Load: librosa.load() → numpy array
- Save: soundfile.write()
- Validate: duration, silence check
- Supports: FLAC, WAV, MP3, M4A, OGG

**AudioRecorder**:
- Real-time microphone capture
- sounddevice.InputStream (float32)
- Background thread for processing
- VAD integration for auto-stop
- Input gain control (amplify weak signals)

#### 4. Preprocessing (`src/preprocessing/`)

**AudioPreprocessor**:
- **Normalize**: Peak normalization (max = 1.0)
- **Trim Silence**: librosa.effects.trim()
- **Denoise**: noisereduce (spectral gating)
- **Resample**: librosa.resample()

**Note**: Preprocessing is **DISABLED** by default in config because Whisper has built-in preprocessing.

**VoiceActivityDetector**:
- Silero VAD model (torch.hub)
- Detect speech segments
- Calculate silence duration
- Auto-stop recording trigger

#### 5. Utilities (`src/utils/`)

**audio_utils.py**:
- `load_audio()` - Load with librosa
- `save_audio()` - Save with soundfile
- `validate_audio()` - Duration + silence check
- `format_timestamp()` - HH:MM:SS.mmm format

**logger_setup.py**:
- Loguru configuration
- File + console logging
- Rotation: 100 MB
- Retention: 1 week

---

## 🧪 Test & Evaluation System

### Test Framework (`tests/evaluation/`)

**ASRBenchmarker** (Main orchestrator):
- Load test set (ground_truth.json)
- Run single test with model config
- Model comparison (tiny, base, small, medium, large-v3-w4a16)
- Implementation comparison (whisper vs faster-whisper)
- Preprocessing comparison

**Metrics** (`metrics.py`):
- **WER** (Word Error Rate): jiwer library
- **CER** (Character Error Rate): Character-level accuracy
- **RTF** (Real-Time Factor): processing_time / audio_duration
- **Normalized metrics**: Turkish text normalization (case, punctuation)

**ResourceMonitor** (`resource_monitor.py`):
- Context manager for resource tracking
- CPU usage: psutil.cpu_percent()
- Memory: psutil.Process().memory_info()
- Peak memory tracking

**ReportGenerator** (`report_generator.py`):
- Export formats: JSON, CSV, Markdown, LaTeX
- Comparison tables
- Summary statistics

### Test Scripts (`tests/scripts/`)

**run_benchmarks.py** (Main runner):
- Modes: full, models, implementations, preprocessing, quick
- Sample limiting
- Multi-format export

**compare_models.py**:
- Quick model comparison
- 50-150 samples
- Table output

**quick_test.py**:
- 5 samples
- Fast validation
- System check

**prepare_test_set.py**:
- Random sample selection from data/raw/TR/
- Generate ground_truth.json
- 300 samples default

---

## 🔍 Data Flow Example: File Upload Transcription

```
1. User uploads "audio.mp3" (3 minutes, Turkish)

2. AudioFileHandler.load()
   Input: "audio.mp3"
   Output: numpy array (2,880,000 samples @ 16kHz), sr=16000

3. Validation
   - Duration: 180s ✓
   - Not silent: max amplitude > 0.02 ✓

4. Model Selection (from config)
   - model.name: "faster-whisper"
   - model.variant: "medium"
   - model.compute_type: "int8"

5. ModelManager.load_model()
   - Factory selects: FasterWhisperASR
   - Load CTranslate2 model: "medium"
   - Device: CPU, compute_type: int8

6. Transcription
   FasterWhisperASR.transcribe(audio, language="tr")
   - Input: numpy array (float32, mono, 16kHz)
   - CTranslate2 inference
   - Beam search (beam_size=5)
   - Generate segments with timestamps

7. Output Formatting
   {
     'text': "Merhaba, bu bir test kaydıdır...",
     'language': "tr",
     'segments': [
       {'start': 0.0, 'end': 2.5, 'text': "Merhaba,"},
       {'start': 2.5, 'end': 5.0, 'text': "bu bir test kaydıdır..."}
     ],
     'processing_time': 45.2,
     'audio_duration': 180.0,
     'rtf': 0.25
   }

8. UI Display
   - Text in expandable card
   - Segments with timestamps
   - Metrics: WER, RTF, duration
   - Download button
```

---

## 🎛️ Configuration System

### Config Hierarchy

```yaml
config.yaml
├─> model:
│   ├─> name: "faster-whisper" | "whisper"
│   ├─> variant: "tiny" | "base" | "small" | "medium" | "large-v3-w4a16"
│   ├─> device: "cpu" | "cuda" | "mps"
│   ├─> compute_type: "int8" | "float16" | "float32"
│   └─> model_path: null | "./path/to/custom/model"
│
├─> audio:
│   ├─> sample_rate: 16000
│   ├─> channels: 1
│   └─> preprocessing:
│       ├─> enabled: false  # Disabled by default
│       ├─> normalize: false
│       ├─> trim_silence: false
│       └─> denoise: false
│
├─> vad:
│   ├─> enabled: true
│   ├─> threshold: 0.5
│   ├─> min_silence_duration_ms: 10000  # 10 seconds
│   └─> model: "silero"
│
├─> transcription:
│   ├─> task: "transcribe"
│   ├─> temperature: 0.0
│   ├─> beam_size: 5
│   └─> word_timestamps: false
│
└─> evaluation:
    ├─> test_set_path: "./tests/data/test_set"
    ├─> model_variants: ["tiny", "base", "small", "medium", "large-v3-w4a16"]
    └─> custom_models:
        └─> large_v3_quantized:
            ├─> name: "whisper"
            ├─> variant: "large-v3-w4a16"
            ├─> model_path: "./checkpoints/hf_models/whisper-large-v3-w4a16"
            └─> device: "cpu"
```

### Runtime Config Updates

```python
from config import config

# Read
variant = config.get('model.variant')  # "base"

# Update
config.set('model.variant', 'small')

# Save to file
config.save()
```

---

## 🚀 Performance Characteristics

### Model Comparison (300 samples, MacBook M4 Pro)

| Model | Implementation | WER | RTF | Memory | Notes |
|-------|---------------|-----|-----|--------|-------|
| Tiny | faster-whisper | 71% | 0.09x | 0.87 GB | Fastest, least accurate |
| Base | faster-whisper | 53% | 0.13x | 0.84 GB | Good balance |
| Small | faster-whisper | 36% | 0.22x | 0.85 GB | Better accuracy |
| Medium | faster-whisper | 27% | 0.39x | 0.86 GB | Recommended |
| Large-v3 INT4 | whisper (quantized) | **19%** | 33.7x | 2.1 GB | Best accuracy, very slow |

**RTF (Real-Time Factor)**:
- 0.1x = 10x faster than real-time
- 1.0x = real-time
- 33.7x = 33.7x slower than real-time

---

## 🔐 Security & Privacy

- **100% Local**: No cloud API calls
- **No Data Collection**: All processing on-device
- **No Network Required**: Models downloaded once, cached locally
- **Open Source**: Full code transparency

---

## 📚 Key Technologies

| Component | Technology | Purpose |
|-----------|-----------|---------|
| ASR Engine | Faster-Whisper (CTranslate2) | Fast inference |
| Quantization | compressed-tensors (INT4) | Model compression |
| VAD | Silero VAD | Speech detection |
| Audio I/O | sounddevice, librosa, soundfile | Recording & file handling |
| UI | Streamlit | Web interface |
| Testing | pytest, jiwer | Unit tests & WER/CER |
| Monitoring | psutil | Resource tracking |
| Config | PyYAML | Configuration management |
| Logging | loguru | Structured logging |

---

## 🎓 For Developers

### Adding a New Model

1. Create new class inheriting from `BaseASR`
2. Implement: `load()`, `transcribe()`, `unload()`
3. Add to `ModelManager._load_*()` factory method
4. Update `ModelType` enum
5. Add config entry in `config.yaml`

### Adding a New Metric

1. Add function to `tests/evaluation/metrics.py`
2. Update `ASRBenchmarker.run_single_test()` to calculate it
3. Add to report templates in `ReportGenerator`

### Debugging Tips

- Check logs: `logs/asr_system.log`
- Enable debug logging: `config.set('logging.level', 'DEBUG')`
- Use `quick_test.py` for fast iteration
- Monitor resources with `ResourceMonitor`

---

## 📊 Project Statistics

- **Total Modules**: 20+ Python modules
- **Lines of Code**: ~5,000 (excluding tests)
- **Test Coverage**: Unit tests + integration benchmarks
- **Supported Languages**: Turkish, English (extensible)
- **Model Variants**: 5 (tiny, base, small, medium, large-v3-w4a16)
- **Test Dataset**: 300 samples (Mozilla Common Voice Turkish)

---

**Last Updated**: December 2025  
**Project**: Gazi University Computer Engineering - ASR System

