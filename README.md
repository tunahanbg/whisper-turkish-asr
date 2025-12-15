# 🎤 Türkçe ve İngilizce Konuşma Tanıma Sistemi
### Local Speech-to-Text System with Turkish and English Support

**Gazi Üniversitesi Bilgisayar Mühendisliği**  
**Tunahan Başaran Güneysu**  
**Aralık 2025**

---

## 📖 Proje Hakkında

Bu proje, MacOS M4 Pro platformunda çalışan, **Türkçe** ve **İngilizce** destekli yerel bir konuşma tanıma (Speech-to-Text) sistemidir. OpenAI Whisper modelini temel alır ve tamamen yerel olarak çalışarak kullanıcı gizliliğini korur.

### ✨ Özellikler

- 🇹🇷 **Türkçe Desteği**: Whisper Medium modelinin Türkçe için fine-tune edilmesi
- 🇬🇧 **İngilizce Desteği**: Yüksek doğruluk oranıyla İngilizce transkripsiyon
- 🎙️ **Gerçek Zamanlı Mikrofon Kaydı**: VAD (Voice Activity Detection) ile otomatik sessizlik algılama
- 📁 **Ses Dosyası Yükleme**: WAV, MP3, M4A formatlarını destekler
- 🖥️ **Streamlit Arayüzü**: Kullanıcı dostu web tabanlı arayüz
- 🔒 **Tamamen Yerel**: Bulut API'lerine ihtiyaç duymaz, verileriniz yerel kalır
- ⚡ **Apple Silicon Optimizasyonu**: M4 Pro üzerinde MPS backend ile hızlandırılmış

### 🎯 Performans Hedefleri

| Dil | Hedef WER | Model |
|-----|-----------|-------|
| 🇹🇷 Türkçe | ≤ %8 (Fine-tune sonrası) | Whisper Medium |
| 🇬🇧 İngilizce | ≤ %5 | Whisper Medium |

---

## 🛠️ Teknoloji Yığını

| Kategori | Teknoloji |
|----------|-----------|
| **ASR Modeli** | OpenAI Whisper Medium |
| **VAD** | Silero VAD |
| **Framework** | PyTorch + MPS (Apple Silicon) |
| **Fine-tuning** | HuggingFace Transformers + PEFT/LoRA |
| **UI** | Streamlit |
| **Ses İşleme** | librosa, pydub, sounddevice |
| **Dataset** | Mozilla Common Voice Turkish |

---

## 📋 Gereksinimler

- **İşletim Sistemi**: macOS (Apple Silicon M1/M2/M3/M4)
- **Python**: 3.11+
- **RAM**: 16GB+ önerilir
- **Disk Alanı**: ~10GB (model ve dataset için)

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

## 📖 Dokümantasyon

Detaylı proje gereksinimleri, mimari tasarım ve geliştirme fazları için:

👉 [Product Requirements Document (PRD)](docs/PRD_Speech_Recognition_TR_EN.md)

---

## 📁 Proje Yapısı

```
ASR_School_Project/
├── src/                    # Kaynak kodlar
│   ├── audio/             # Ses yakalama ve işleme
│   ├── models/            # Model yükleme ve inference
│   ├── preprocessing/     # Ses ön işleme pipeline'ı
│   └── ui/                # Streamlit arayüzü
├── data/                  # Dataset ve örnek dosyalar
├── tests/                 # Test dosyaları
├── notebooks/             # Jupyter notebook'lar
├── docs/                  # Dokümantasyon
├── requirements.txt       # Python bağımlılıkları
└── README.md
```

---

## 🎯 Geliştirme Fazları

- [x] **Faz 0**: Proje dokümantasyonu (PRD)
- [ ] **Faz 1**: Ortam kurulumu ve temel altyapı
- [ ] **Faz 2**: Temel ASR işlevselliği (Whisper Medium)
- [ ] **Faz 3**: Mikrofon entegrasyonu ve VAD
- [ ] **Faz 4**: Ses ön işleme pipeline'ı
- [ ] **Faz 5**: Dataset hazırlığı ve fine-tuning
- [ ] **Faz 6**: Streamlit arayüzü
- [ ] **Faz 7**: Test ve değerlendirme
- [ ] **Faz 8**: Dokümantasyon ve rapor

---

## 📊 Kullanılan Dataset

- **Mozilla Common Voice Turkish v17.0**
  - 134 saat Türkçe ses kaydı
  - 1,790 benzersiz konuşmacı
  - Creative Commons Zero (CC-0) lisanslı

---

## 📝 Lisans

Bu proje akademik amaçlarla geliştirilmiştir.

---

## 👤 İletişim

**Tunahan Başaran Güneysu**  
Gazi Üniversitesi - Bilgisayar Mühendisliği  
GitHub: [@tunahanbg](https://github.com/tunahanbg)

---

## 🙏 Teşekkürler

- OpenAI Whisper ekibine
- Mozilla Common Voice topluluğuna
- Silero VAD geliştiricilerine
- Gazi Üniversitesi Bilgisayar Mühendisliği Bölümü'ne

---

**⭐ Bu projeyi beğendiyseniz, yıldız vermeyi unutmayın!**

