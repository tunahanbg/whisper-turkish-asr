# ASR System - Development Guide

## 📁 Proje Yapısı

```
ASR_School_Project/
├── config/                 # Konfigürasyon dosyaları
│   ├── __init__.py        # Config manager (Singleton pattern)
│   └── config.yaml        # Ana konfigürasyon dosyası
│
├── src/                   # Kaynak kodlar
│   ├── __init__.py       # Proje root initialization
│   │
│   ├── audio/            # Ses yakalama ve dosya işlemleri
│   │   ├── __init__.py
│   │   ├── recorder.py   # Mikrofon kaydı (VAD destekli)
│   │   └── file_handler.py  # Dosya yükleme/kaydetme
│   │
│   ├── models/           # Model yönetimi
│   │   ├── __init__.py
│   │   ├── whisper_model.py    # Whisper wrapper
│   │   └── model_manager.py    # Factory pattern model yönetimi
│   │
│   ├── preprocessing/    # Ses ön işleme
│   │   ├── __init__.py
│   │   ├── processor.py  # Audio preprocessing pipeline
│   │   └── vad.py        # Voice Activity Detection (Silero)
│   │
│   ├── ui/               # Streamlit UI
│   │   ├── __init__.py
│   │   └── app.py        # Ana Streamlit uygulaması
│   │
│   └── utils/            # Utility fonksiyonlar
│       ├── __init__.py
│       ├── logger_setup.py   # Logging konfigürasyonu
│       └── audio_utils.py    # Audio utility fonksiyonları
│
├── tests/                # Test dosyaları
│   ├── __init__.py
│   ├── test_config.py
│   ├── test_audio_utils.py
│   └── test_preprocessor.py
│
├── examples/             # Kullanım örnekleri
│   └── basic_usage.py
│
├── data/                 # Data dizini
│   ├── raw/             # Ham veriler
│   ├── processed/       # İşlenmiş veriler
│   ├── cache/           # Cache dosyaları
│   └── examples/        # Örnek ses dosyaları
│
├── checkpoints/         # Model checkpoints
├── logs/                # Log dosyaları
├── notebooks/           # Jupyter notebooks
│
├── app.py               # Streamlit entry point
├── requirements.txt     # Python dependencies
├── .gitignore
└── README.md
```

## 🎯 Tasarım Prensipleri

### 1. DRY (Don't Repeat Yourself)
- Tüm tekrarlanan kod merkezi modüllere taşındı
- Utility fonksiyonlar tek bir yerde tanımlandı
- Config-driven approach ile parametreler merkezi yönetiliyor

### 2. Modülerlik
- Her modül tek bir sorumluluğa sahip (Single Responsibility Principle)
- Modüller birbirinden bağımsız çalışabilir
- Factory pattern ile model değiştirme kolaylığı

### 3. Config-Driven Design
- Tüm parametreler `config/config.yaml` dosyasında
- Model değişikliği için sadece config güncellemek yeterli
- Runtime'da config değişiklikleri mümkün

### 4. Singleton Pattern
- Config class singleton olarak implement edildi
- Tüm uygulama tek bir config instance kullanır
- Thread-safe config erişimi

## 🔧 Kullanım

### 1. Environment Kurulumu

```bash
# Virtual environment aktif et
source asr_project/bin/activate

# Kütüphaneler yüklü mü kontrol et
pip list | grep whisper
```

### 2. Streamlit UI Çalıştırma

```bash
streamlit run app.py
```

### 3. Programatik Kullanım

```python
from src.models import ModelManager
from src.audio import AudioFileHandler
from src.preprocessing import AudioPreprocessor

# Model yükle
model_manager = ModelManager()
model_manager.load_model()
model = model_manager.get_model()

# Ses dosyası yükle
handler = AudioFileHandler()
audio, sr = handler.load("example.wav")

# Ön işleme
preprocessor = AudioPreprocessor()
processed_audio = preprocessor.process(audio)

# Transkripsiyon
result = model.transcribe(processed_audio, language='tr')
print(result['text'])
```

### 4. Config Özelleştirme

```python
from config import config

# Mevcut ayarları görüntüle
print(config.get('model.variant'))  # 'medium'

# Runtime'da güncelle
config.set('model.variant', 'small')

# Dosyaya kaydet
config.save()
```

## 📝 Test Çalıştırma

```bash
# Tüm testler
pytest tests/

# Specific test file
pytest tests/test_config.py

# Coverage ile
pytest --cov=src tests/
```

## 🎯 Örnek Kullanımlar

```bash
# Basic usage examples
python examples/basic_usage.py
```

## 🔍 Önemli Modüller

### Config Manager (`config/__init__.py`)
- Singleton pattern ile merkezi config yönetimi
- Nested key access: `config.get('model.variant')`
- Runtime updates: `config.set('key', value)`

### Model Manager (`src/models/model_manager.py`)
- Factory pattern ile farklı model tipleri desteği
- Lazy loading: Model sadece gerektiğinde yüklenir
- Easy model switching

### Audio Preprocessor (`src/preprocessing/processor.py`)
- Modüler preprocessing pipeline
- Her adım config'den kontrol edilebilir
- Normalize, trim silence, denoise

### VAD (`src/preprocessing/vad.py`)
- Silero VAD entegrasyonu
- Otomatik sessizlik algılama
- Recording auto-stop

### Audio Recorder (`src/audio/recorder.py`)
- Real-time mikrofon kaydı
- VAD ile otomatik durdurma
- Background thread processing

## 🚀 Next Steps

1. **Model Fine-tuning**: Türkçe dataset ile model optimize edilecek
2. **Faster-Whisper**: Optimized inference için entegre edilecek
3. **MLX Whisper**: Apple Silicon native implementation
4. **Real-time Streaming**: Streaming transcription desteği
5. **Evaluation Pipeline**: WER/CER hesaplama modülü

## 📚 Kaynaklar

- [Whisper Documentation](https://github.com/openai/whisper)
- [Silero VAD](https://github.com/snakers4/silero-vad)
- [Streamlit Docs](https://docs.streamlit.io)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)

