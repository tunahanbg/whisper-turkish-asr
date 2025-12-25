# 🎤 Türkçe ve İngilizce Konuşma Tanıma Sistemi

### Local Speech-to-Text System with Turkish and English Support

---

## 📖 Proje Hakkında

Bu proje, MacOS M4 Pro platformunda çalışan, **Türkçe** ve **İngilizce** destekli yerel bir konuşma tanıma (Speech-to-Text) sistemidir. OpenAI Whisper modelini temel alır ve tamamen yerel olarak çalışarak kullanıcı gizliliğini korur.

### ✨ Özellikler

-   🇹🇷 **Türkçe Desteği**: OpenAI Whisper modelleri ile Türkçe transkripsiyon
-   🇬🇧 **İngilizce Desteği**: Çok dilli model desteği
-   🎙️ **Gerçek Zamanlı Mikrofon Kaydı**: VAD (Voice Activity Detection) ile otomatik sessizlik algılama
-   📁 **Ses Dosyası Yükleme**: FLAC, WAV, MP3, M4A formatlarını destekler
-   🖥️ **Streamlit Arayüzü**: Kullanıcı dostu web tabanlı arayüz
-   🔒 **Tamamen Yerel**: Bulut API'lerine ihtiyaç duymaz, verileriniz yerel kalır
-   ⚡ **Faster-Whisper**: CTranslate2 backend ile hızlandırılmış inference
-   🎯 **Quantized Model**: INT4 quantized large-v3 model desteği (daha doğru, yavaş)

### 🎯 Benchmark Sonuçları (300 Örnek)

| Model                   | WER (Normalized) | RTF    | CPU | Memory  |
| ----------------------- | ---------------- | ------ | --- | ------- |
| Faster-Whisper Tiny     | 71.09%           | 0.093x | 38% | 0.87 GB |
| Faster-Whisper Base     | 52.69%           | 0.127x | 45% | 0.84 GB |
| Faster-Whisper Small    | 35.60%           | 0.218x | 53% | 0.85 GB |
| Faster-Whisper Medium   | 27.41%           | 0.389x | 62% | 0.86 GB |
| Large-v3 INT4 Quantized | **18.96%**       | 33.7x  | 92% | 2.1 GB  |

> **RTF**: Real-Time Factor (1.0x = gerçek zamanlı)  
> Quantized model en doğru ama CPU'da çok yavaş

---

## 🛠️ Teknoloji Yığını

| Kategori         | Teknoloji                                         |
| ---------------- | ------------------------------------------------- |
| **ASR Modeli**   | Faster-Whisper (CTranslate2) + Quantized Large-v3 |
| **VAD**          | Silero VAD                                        |
| **Framework**    | PyTorch + HuggingFace Transformers                |
| **Quantization** | INT4 (compressed-tensors)                         |
| **UI**           | Streamlit + streamlit-webrtc                      |
| **Ses İşleme**   | librosa, sounddevice, pydub                       |
| **Test Dataset** | Mozilla Common Voice Turkish (300 samples)        |
| **Evaluation**   | WER/CER metrics, resource monitoring              |

---

## 📋 Gereksinimler

-   **İşletim Sistemi**: macOS (Apple Silicon M1/M2/M3/M4)
-   **Python**: 3.11+
-   **RAM**: 16GB+ önerilir
-   **Disk Alanı**: ~10GB (model ve dataset için)

---

## 🚀 Kurulum

### 1. Repoyu Klonlayın

```bash
git clone https://github.com/tunahanbg/ASR_School_Project.git
cd ASR_School_Project
```

### 2. Sanal Ortam Oluşturun

```bash
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
```

### 3. Bağımlılıkları Yükleyin

```bash
pip install -r requirements.txt
```

### 4. PyTorch MPS Desteğini Kontrol Edin

```python
import torch
print(torch.backends.mps.is_available())  # True olmalı
```

---

## 💻 Kullanım

### Streamlit UI Başlatma

```bash
cd src/ui
streamlit run app.py
```

Tarayıcınızda `http://localhost:8501` adresine gidin.

### Model Seçimi

UI'da iki model tipi mevcuttur:

1. **Standard (Faster-Whisper)**: Hızlı, CPU-verimli

    - Tiny, Base, Small, Medium, Large seçenekleri
    - Önerilen: Medium (WER: %27, RTF: 0.39x)

2. **Quantized Large (INT4)**: En doğru, yavaş
    - WER: %19 (en iyi doğruluk)
    - RTF: 33.7x (çok yavaş, CPU-bound)

### Benchmark Çalıştırma

```bash
# Hızlı test (5 sample)
python tests/scripts/quick_test.py

# Model karşılaştırma (varsayılan: 150 sample)
python tests/scripts/compare_models.py --samples 150 --save

# Detaylı benchmark
python tests/scripts/run_benchmarks.py --mode full
```

Detaylar: [BENCHMARK_GUIDE.md](docs/BENCHMARK_GUIDE.md)

---

## 📖 Dokümantasyon

Detaylı proje gereksinimleri, mimari tasarım ve geliştirme fazları için:

👉 **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** - Sistem mimarisi ve modül akışı (BAŞLANGIÇ NOKTASI)  
👉 [Product Requirements Document (PRD)](docs/PRD_Speech_Recognition_TR_EN.md) - Proje gereksinimleri  
👉 [DEVELOPMENT.md](DEVELOPMENT.md) - Geliştirme kılavuzu  
👉 [BENCHMARK_GUIDE.md](docs/BENCHMARK_GUIDE.md) - Benchmark kullanım kılavuzu  
👉 [CLEANUP_SUMMARY.md](docs/CLEANUP_SUMMARY.md) - Proje temizleme raporu  
👉 [Test System Documentation](tests/README.md) - Test sistemi detayları

---

## 📁 Proje Yapısı

```
ASR_School_Project/
├── app.py                 # Streamlit entry point
├── config/                # Konfigürasyon dosyaları
│   ├── __init__.py       # Config manager (Singleton pattern)
│   └── config.yaml       # Ana konfigürasyon
├── src/                   # Kaynak kodlar
│   ├── audio/            # Ses yakalama ve dosya işleme
│   │   ├── file_handler.py  # Dosya yükleme/kaydetme
│   │   └── recorder.py       # Mikrofon kaydı (VAD destekli)
│   ├── models/           # Model yönetimi
│   │   ├── base_asr.py       # Abstract base class
│   │   ├── model_manager.py  # Factory pattern
│   │   ├── faster_whisper_model.py  # CTranslate2 (primary)
│   │   └── whisper_model.py  # Standard + Quantized HF models
│   ├── preprocessing/    # VAD ve ses ön işleme
│   │   ├── processor.py      # Audio preprocessing pipeline
│   │   └── vad.py            # Silero VAD
│   ├── ui/               # Streamlit arayüzü
│   │   └── app.py            # Ana UI
│   └── utils/            # Utility fonksiyonlar
│       ├── audio_utils.py    # Audio utilities
│       └── logger_setup.py   # Logging config
├── tests/                # Test ve evaluation
│   ├── data/             # Test seti (300 samples) ve sonuçlar
│   ├── evaluation/       # Benchmarking framework
│   │   ├── benchmarker.py    # Ana benchmark modülü
│   │   ├── metrics.py        # WER/CER hesaplama
│   │   ├── resource_monitor.py  # CPU/Memory monitoring
│   │   └── report_generator.py  # Rapor oluşturma
│   └── scripts/          # Benchmark scriptleri
│       ├── run_benchmarks.py   # Ana benchmark runner
│       ├── compare_models.py   # Model karşılaştırma
│       ├── quick_test.py       # Hızlı test
│       └── prepare_test_set.py # Test seti hazırlama
├── examples/             # Kullanım örnekleri
│   └── basic_usage.py    # Programatik kullanım örnekleri
├── data/                 # Dataset
│   ├── raw/TR/          # 300 FLAC samples + transcripts
│   └── cache/           # Geçici dosyalar
├── checkpoints/          # Model checkpoints
│   ├── models--Systran--faster-whisper-*/  # Faster-Whisper models
│   └── quantized_models/whisper-large-v3-w4a16/  # Quantized model
├── docs/                 # Proje dokümantasyonu
│   ├── ARCHITECTURE.md       # Sistem mimarisi (START HERE)
│   ├── BENCHMARK_GUIDE.md    # Benchmark kılavuzu
│   ├── CLEANUP_SUMMARY.md    # Temizleme raporu
│   ├── PRD_Speech_Recognition_TR_EN.md  # Gereksinimler
│   └── FASTER_WHISPER_GUIDE.md          # Implementation guide
├── logs/                 # Log dosyaları
├── requirements.txt      # Python bağımlılıkları
├── DEVELOPMENT.md        # Geliştirme kılavuzu
└── README.md
```

---

## 🎯 Geliştirme Fazları

-   [x] **Faz 0**: Proje dokümantasyonu (PRD)
-   [x] **Faz 1**: Ortam kurulumu ve temel altyapı
-   [x] **Faz 2**: Temel ASR işlevselliği (Faster-Whisper)
-   [x] **Faz 3**: Mikrofon entegrasyonu ve VAD
-   [x] **Faz 4**: Ses ön işleme pipeline'ı
-   [x] **Faz 5**: Quantized model entegrasyonu
-   [x] **Faz 6**: Streamlit arayüzü
-   [x] **Faz 7**: Test ve değerlendirme (300 sample benchmark)
-   [ ] **Faz 8**: Dokümantasyon ve rapor

---

## 📊 Kullanılan Dataset

-   **Mozilla Common Voice Turkish v17.0**
    -   134 saat Türkçe ses kaydı
    -   1,790 benzersiz konuşmacı
    -   Creative Commons Zero (CC-0) lisanslı

---

## 📝 Lisans

Bu proje akademik amaçlarla geliştirilmiştir.

---
