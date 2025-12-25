# Turkish and English Speech Recognition System

Local Speech-to-Text System with Turkish and English Support

---

## About

This project is a local speech recognition (Speech-to-Text) system for **Turkish** and **English** languages, running on MacOS M4 Pro. Built on OpenAI Whisper models, it operates completely locally to ensure user privacy.

### Features

- Turkish language support with OpenAI Whisper models
- English language support with multilingual models
- Real-time microphone recording with VAD (Voice Activity Detection)
- Audio file upload supporting FLAC, WAV, MP3, M4A formats
- Streamlit web interface
- Fully local processing without cloud API dependencies
- Faster-Whisper with CTranslate2 backend for optimized inference
- INT4 quantized large-v3 model support for higher accuracy

### Benchmark Results (300 Samples)

| Model                   | WER (Normalized) | RTF    | CPU | Memory  |
| ----------------------- | ---------------- | ------ | --- | ------- |
| Faster-Whisper Tiny     | 71.09%           | 0.093x | 38% | 0.87 GB |
| Faster-Whisper Base     | 52.69%           | 0.127x | 45% | 0.84 GB |
| Faster-Whisper Small    | 35.60%           | 0.218x | 53% | 0.85 GB |
| Faster-Whisper Medium   | 27.41%           | 0.389x | 62% | 0.86 GB |
| Large-v3 INT4 Quantized | 18.96%           | 33.7x  | 92% | 2.1 GB  |

RTF: Real-Time Factor (1.0x = real-time)

---

## Technology Stack

| Category         | Technology                                        |
| ---------------- | ------------------------------------------------- |
| ASR Model        | Faster-Whisper (CTranslate2) + Quantized Large-v3 |
| VAD              | Silero VAD                                        |
| Framework        | PyTorch + HuggingFace Transformers                |
| Quantization     | INT4 (compressed-tensors)                         |
| UI               | Streamlit + streamlit-webrtc                      |
| Audio Processing | librosa, sounddevice, pydub                       |
| Test Dataset     | Mozilla Common Voice Turkish (300 samples)        |
| Evaluation       | WER/CER metrics, resource monitoring              |

---

## Requirements

- Operating System: macOS (Apple Silicon M1/M2/M3/M4)
- Python: 3.11+
- RAM: 16GB+ recommended
- Disk Space: ~10GB (for models and dataset)

---

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/tunahanbg/whisper-turkish-asr.git
cd whisper-turkish-asr
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify PyTorch MPS Support

```python
import torch
print(torch.backends.mps.is_available())  # Should return True
```

---

## Usage

### Launch Streamlit UI

```bash
streamlit run app.py
```

Navigate to `http://localhost:8501` in your browser.

### Model Selection

Two model types available in the UI:

1. **Standard (Faster-Whisper)**: Fast, CPU-efficient
   - Options: Tiny, Base, Small, Medium, Large
   - Recommended: Medium (WER: 27%, RTF: 0.39x)

2. **Quantized Large (INT4)**: Most accurate, slower
   - WER: 19% (best accuracy)
   - RTF: 33.7x (very slow, CPU-bound)

### Running Benchmarks

```bash
# Quick test (5 samples)
python tests/scripts/quick_test.py

# Model comparison (default: 150 samples)
python tests/scripts/compare_models.py --samples 150 --save

# Full benchmark
python tests/scripts/run_benchmarks.py --mode full
```

---

## Project Structure

```
whisper-turkish-asr/
├── app.py                 # Streamlit entry point
├── config/                # Configuration files
│   ├── __init__.py       # Config manager (Singleton pattern)
│   └── config.yaml       # Main configuration
├── src/                   # Source code
│   ├── audio/            # Audio capture and file handling
│   │   ├── file_handler.py
│   │   └── recorder.py
│   ├── models/           # Model management
│   │   ├── base_asr.py
│   │   ├── model_manager.py
│   │   ├── faster_whisper_model.py
│   │   └── whisper_model.py
│   ├── preprocessing/    # VAD and audio preprocessing
│   │   ├── processor.py
│   │   └── vad.py
│   ├── ui/               # Streamlit interface
│   │   └── app.py
│   └── utils/            # Utility functions
│       ├── audio_utils.py
│       └── logger_setup.py
├── tests/                # Testing and evaluation
│   ├── data/             # Test set (300 samples) and results
│   ├── evaluation/       # Benchmarking framework
│   │   ├── benchmarker.py
│   │   ├── metrics.py
│   │   ├── resource_monitor.py
│   │   └── report_generator.py
│   └── scripts/          # Benchmark scripts
│       ├── run_benchmarks.py
│       ├── compare_models.py
│       ├── quick_test.py
│       └── prepare_test_set.py
├── examples/             # Usage examples
│   └── basic_usage.py
├── data/                 # Dataset
│   ├── raw/TR/          # 300 FLAC samples + transcripts
│   └── cache/           # Temporary files
├── checkpoints/          # Model checkpoints
│   ├── models--Systran--faster-whisper-*
│   └── quantized_models/whisper-large-v3-w4a16
├── logs/                 # Log files
├── requirements.txt      # Python dependencies
└── README.md
```

---

## Development Phases

- [x] Phase 0: Project documentation (PRD)
- [x] Phase 1: Environment setup and infrastructure
- [x] Phase 2: Core ASR functionality (Faster-Whisper)
- [x] Phase 3: Microphone integration and VAD
- [x] Phase 4: Audio preprocessing pipeline
- [x] Phase 5: Quantized model integration
- [x] Phase 6: Streamlit interface
- [x] Phase 7: Testing and evaluation (300 sample benchmark)
- [x] Phase 8: Documentation and reporting

---

## Dataset

**Mozilla Common Voice Turkish v17.0**
- 134 hours of Turkish audio
- 1,790 unique speakers
- Licensed under Creative Commons Zero (CC-0)

---

## License

This project is developed for academic purposes.

---
