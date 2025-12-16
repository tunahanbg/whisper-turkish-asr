# Product Requirements Document (PRD)

## Türkçe ve İngilizce Destekli Yerel Konuşma Tanıma Sistemi
### Local Speech-to-Text System with Turkish and English Support

**Hazırlayan:** Tunahan  
**Kurum:** Gazi Üniversitesi Bilgisayar Mühendisliği  
**Tarih:** Aralık 2025

---

## Özet (Abstract)

Bu doküman, MacOS M4 Pro platformunda çalışacak, Türkçe ve İngilizce destekli yerel bir konuşma tanıma (Speech-to-Text) sistemi için Ürün Gereksinimleri Dokümanı'nı (PRD) sunmaktadır. Sistem, OpenAI Whisper modelini temel alacak, Streamlit tabanlı bir kullanıcı arayüzü sunacak ve Voice Activity Detection (VAD) ile otomatik sessizlik algılama özelliği içerecektir. Proje, hazır modelin fine-tuning ile Türkçe için optimize edilmesini, ses ön işleme pipeline'ının oluşturulmasını ve Word Error Rate (WER) metriği ile performans değerlendirmesini kapsamaktadır.

---

## İçindekiler

1. [Giriş ve Proje Amacı](#1-giriş-ve-proje-amacı)
2. [Model Seçimi ve Gerekçelendirme](#2-model-seçimi-ve-gerekçelendirme)
3. [Dataset Seçimi](#3-dataset-seçimi)
4. [Performans Hedefleri](#4-performans-hedefleri)
5. [Teknik Mimari](#5-teknik-mimari)
6. [Geliştirme Fazları ve Kontrol Listeleri](#6-geliştirme-fazları-ve-kontrol-listeleri)
7. [Teknoloji Yığını Özeti](#7-teknoloji-yığını-özeti)
8. [Kaynaklar](#8-kaynaklar)

---

## 1. Giriş ve Proje Amacı

Bu proje, mikrofon girdisini metne dönüştüren yerel bir konuşma tanıma sistemi geliştirmeyi amaçlamaktadır. Sistem, MacBook Pro M4 Pro üzerinde tamamen yerel olarak çalışacak, bulut tabanlı API'lere ihtiyaç duymayacak ve kullanıcı gizliliğini koruyacaktır.

### 1.1 Proje Kapsamı

- Türkçe ve İngilizce dil desteği
- Whisper modelinin fine-tuning ile Türkçe optimizasyonu
- VAD (Voice Activity Detection) ile sessizlik tespitiyle otomatik kayıt durdurma
- Streamlit tabanlı kullanıcı arayüzü
- Ses dosyası yükleme desteği
- Ses ön işleme pipeline'ı (preprocessing)

### 1.2 Hedef Platform

| Özellik | Değer |
|---------|-------|
| **Cihaz** | MacBook Pro M4 Pro |
| **İşletim Sistemi** | macOS |
| **Çalışma Modu** | Tamamen yerel (offline capable) |
| **RAM Gereksinimi** | 16GB+ önerilir |

---

## 2. Model Seçimi ve Gerekçelendirme

### 2.1 Seçilen Model: Whisper Medium

Apple Silicon M4 Pro için yapılan benchmark analizleri ve Türkçe ASR literatürü değerlendirmesi sonucunda **Whisper Medium** modeli optimal seçenek olarak belirlenmiştir.

### 2.2 Whisper Model Karşılaştırması

| Model | Parametre | VRAM | Göreceli Hız | Türkçe WER (Baseline) |
|-------|-----------|------|--------------|----------------------|
| Tiny | 39M | ~1GB | ~32x | ~25-30% |
| Base | 74M | ~1GB | ~16x | ~20-25% |
| Small | 244M | ~2GB | ~6x | ~14-18% |
| **Medium** | **769M** | **~5GB** | **~2x** | **~8-12%** |
| Large | 1550M | ~10GB | ~1x | ~6-10% |

### 2.3 M4 Pro Performans Özellikleri

Apple Silicon M4 Pro üzerinde Whisper modelleri için:

- **Metal Performance Shaders (MPS)** desteği ile GPU hızlandırma
- **Unified Memory Architecture** sayesinde verimli bellek kullanımı
- **CoreML** entegrasyonu ile 2-3x ek hız artışı mümkün
- **MLX Framework** ile native Apple Silicon optimizasyonu

### 2.4 Model Seçim Gerekçeleri

1. **Yüksek Doğruluk:** Medium model, Türkçe gibi aglütinatif dillerde Small'a göre önemli ölçüde daha düşük WER sağlar

2. **Fine-tuning Kapasitesi:** Daha fazla parametre ile fine-tuning sonrası daha iyi sonuçlar elde edilebilir

3. **M4 Pro Uyumluluğu:** 
   - M4 Pro'nun yüksek bellek bant genişliği Medium model için yeterli
   - Unified Memory yapısı ~5GB VRAM gereksinimini karşılar
   - Gerçek zamanlı işleme için yeterli hız (~2x real-time)

4. **Çoklu Dil Desteği:** Türkçe ve İngilizce için tek model kullanımı mümkün

5. **Literatür Desteği:** Türkçe ASR çalışmalarında Medium model en iyi maliyet-fayda oranını sunmaktadır

### 2.5 Türkçe WER Performansı (Literatür)

Araştırmalara göre Whisper modellerinin Türkçe performansı:

| Durum | WER Aralığı | Kaynak |
|-------|-------------|--------|
| Whisper Medium (baseline) | %8-14 | MDPI Electronics 2024 |
| Whisper Medium (fine-tuned) | %4-8 | LoRA ile fine-tuning |
| Fine-tuning ile iyileşme | %30-52 | Çeşitli çalışmalar |

---

## 3. Dataset Seçimi

### 3.1 Ana Dataset: Mozilla Common Voice Türkçe

| Özellik | Değer |
|---------|-------|
| **Toplam Süre** | 134 saat (129 saat doğrulanmış) |
| **Konuşmacı Sayısı** | 1,790 benzersiz konuşmacı |
| **Ortalama Kayıt Süresi** | ~3.8 saniye |
| **Lisans** | Creative Commons Zero (CC-0) |
| **Erişim** | HuggingFace / Mozilla Data Collective |
| **Versiyon** | Common Voice 23.0 |

**İndirme:**
```python
from datasets import load_dataset

dataset = load_dataset("mozilla-foundation/common_voice_17_0", "tr", split="train")
```

### 3.2 Ek Datasetler (Opsiyonel)

| Dataset | Süre | Açıklama |
|---------|------|----------|
| **ISSAI Turkish Speech Corpus** | 218.2 saat | En büyük Türkçe açık kaynak dataset |
| **MediaSpeech Turkish** | 10 saat | Medya kayıtları (OpenSLR) |
| **FLEURS Turkish** | ~10 saat | Google çok dilli dataset |

### 3.3 Dataset Hazırlık Stratejisi

1. **Train/Validation/Test Bölümlemesi:** 80/10/10
2. **Veri Temizliği:** Düşük kaliteli kayıtların filtrelenmesi
3. **Augmentation (Opsiyonel):** Gürültü ekleme, hız değişimi
4. **Format Dönüşümü:** 16kHz, mono, WAV formatı

---

## 4. Performans Hedefleri

### 4.1 WER (Word Error Rate) Hedefleri

Literatürdeki başarılı çalışmalar ve erişilebilir benchmark değerleri baz alınarak belirlenen hedefler:

| Dil | Hedef WER | Baseline WER | Referans |
|-----|-----------|--------------|----------|
| **Türkçe** | ≤ %8 (Fine-tune sonrası) | %12-14 | Wav2Vec2 TR: %10.61 |
| **İngilizce** | ≤ %5 | %4-5 | Whisper baseline |

### 4.2 Performans Metrikleri

```
WER = (S + D + I) / N × 100

S = Substitution (yanlış kelime)
D = Deletion (eksik kelime)
I = Insertion (fazla kelime)
N = Referanstaki toplam kelime sayısı
```

### 4.3 Ek Metrikler

- **CER (Character Error Rate):** Karakter bazlı hata oranı
- **Real-time Factor (RTF):** İşlem süresi / Ses süresi
- **Latency:** İlk çıktıya kadar geçen süre

---

## 5. Teknik Mimari

### 5.1 Sistem Bileşenleri

```
┌─────────────────────────────────────────────────────────────┐
│                     STREAMLIT UI                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  Mikrofon    │  │  Dosya       │  │  Sonuç       │      │
│  │  Kaydı       │  │  Yükleme     │  │  Gösterimi   │      │
│  └──────┬───────┘  └──────┬───────┘  └──────────────┘      │
└─────────┼─────────────────┼─────────────────────────────────┘
          │                 │
          ▼                 ▼
┌─────────────────────────────────────────────────────────────┐
│                    SES YAKALAMA MODÜLÜ                       │
│         (sounddevice / PyAudio + Silero VAD)                │
│                                                              │
│  • 16kHz sample rate                                         │
│  • 10-15 sn sessizlik algılama                              │
│  • Otomatik kayıt durdurma                                  │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  ÖN İŞLEME PIPELINE'I                        │
│                                                              │
│  1. Resampling (16kHz)                                      │
│  2. Mono dönüşüm                                            │
│  3. Normalizasyon                                           │
│  4. Sessizlik kırpma                                        │
│  5. Gürültü azaltma (opsiyonel)                            │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    WHISPER MEDIUM                            │
│              (Fine-tuned for Turkish)                        │
│                                                              │
│  • PyTorch + MPS Backend                                    │
│  • Otomatik dil algılama                                    │
│  • Timestamp desteği                                        │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
                        [Transcript]
```

### 5.2 VAD (Voice Activity Detection) Yapılandırması

**Seçilen Çözüm: Silero VAD**

| Özellik | Değer |
|---------|-------|
| Model Boyutu | ~2 MB |
| İşlem Hızı | 30ms chunk < 1ms CPU |
| Dil Desteği | 100+ dil (Türkçe dahil) |
| Sample Rate | 8kHz ve 16kHz |
| Sessizlik Eşiği | 10-15 saniye (yapılandırılabilir) |
| Lisans | MIT |

**Kullanım Örneği:**
```python
import torch

model, utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad'
)

(get_speech_timestamps, _, read_audio, _, _) = utils

wav = read_audio('audio.wav', sampling_rate=16000)
speech_timestamps = get_speech_timestamps(
    wav, 
    model, 
    sampling_rate=16000,
    threshold=0.5,
    min_silence_duration_ms=10000  # 10 saniye
)
```

### 5.3 Ses Ön İşleme Pipeline'ı

Whisper zaten gürültüye dayanıklı olarak eğitilmiş olsa da, aşağıdaki ön işleme adımları WER'de iyileştirme sağlayabilir:

#### 5.3.1 Temel Ön İşleme (Zorunlu)

```python
import librosa
import numpy as np

def basic_preprocessing(audio_path):
    # 1. Yükleme ve resampling
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    
    # 2. Normalizasyon
    audio = audio / np.max(np.abs(audio))
    
    # 3. Sessizlik kırpma
    audio, _ = librosa.effects.trim(audio, top_db=20)
    
    return audio
```

#### 5.3.2 Gelişmiş Ön İşleme (Gürültülü Ortamlar İçin)

```python
import noisereduce as nr

def advanced_preprocessing(audio, sr=16000):
    # Spectral gating ile gürültü azaltma
    audio_denoised = nr.reduce_noise(
        y=audio, 
        sr=sr,
        prop_decrease=0.8
    )
    return audio_denoised
```

> ⚠️ **Önemli Not:** Whisper gürültülü verilerle eğitildiği için aşırı ön işleme spektral bozulmaya yol açabilir. Ön işleme modüler olarak tasarlanmalı ve performans testleri yapılmalıdır.

---

## 6. Geliştirme Fazları ve Kontrol Listeleri

### Faz 1: Ortam Kurulumu ve Temel Altyapı

**Amaç:** Geliştirme ortamının hazırlanması ve temel bağımlılıkların kurulumu

**Teknolojiler:**
- Python 3.11+
- PyTorch (Apple Silicon MPS desteği)
- Whisper (openai-whisper)
- Streamlit
- librosa, sounddevice, pydub

**Kontrol Listesi:**
- [ ] Python 3.11+ kurulumu ve sanal ortam (venv/conda) oluşturma
- [ ] PyTorch MPS backend doğrulaması
  ```python
  import torch
  print(torch.backends.mps.is_available())  # True olmalı
  ```
- [ ] Whisper kütüphanesi kurulumu ve test
  ```bash
  pip install openai-whisper
  ```
- [ ] Streamlit kurulumu
- [ ] Ses işleme kütüphaneleri (librosa, pydub, sounddevice)
- [ ] Proje dizin yapısının oluşturulması
  ```
  project/
  ├── src/
  │   ├── audio/
  │   ├── models/
  │   ├── preprocessing/
  │   └── ui/
  ├── data/
  ├── tests/
  ├── notebooks/
  ├── requirements.txt
  └── README.md
  ```
- [ ] requirements.txt dosyasının hazırlanması
- [ ] Baseline Whisper Medium ile İngilizce test

---

### Faz 2: Temel ASR İşlevselliği

**Amaç:** Whisper modeli ile temel transkripsiyon işlevinin gerçeklenmesi

**Teknolojiler:**
- Whisper Medium model
- faster-whisper (opsiyonel optimizasyon)
- MLX Whisper (Apple Silicon native - opsiyonel)

**Kontrol Listesi:**
- [ ] Whisper Medium modelinin yüklenmesi
  ```python
  import whisper
  model = whisper.load_model("medium")
  ```
- [ ] Ses dosyası yükleme fonksiyonunun yazılması
- [ ] Transkripsiyon fonksiyonunun implementasyonu
  ```python
  def transcribe(audio_path, language=None):
      result = model.transcribe(
          audio_path,
          language=language,
          task="transcribe"
      )
      return result["text"]
  ```
- [ ] Dil algılama özelliğinin test edilmesi
- [ ] Türkçe ve İngilizce örnek dosyalarla test
- [ ] Baseline WER ölçümü
- [ ] İşlem süresi benchmark (RTF hesaplama)

---

### Faz 3: Mikrofon Entegrasyonu ve VAD

**Amaç:** Gerçek zamanlı mikrofon kaydı ve VAD ile otomatik durdurma

**Teknolojiler:**
- sounddevice / PyAudio
- Silero VAD (PyTorch/ONNX)
- threading / asyncio

**Kontrol Listesi:**
- [ ] Mikrofon erişim izinlerinin ayarlanması (macOS)
  ```bash
  # System Preferences > Security & Privacy > Microphone
  ```
- [ ] Ses yakalama modülünün implementasyonu
  ```python
  import sounddevice as sd
  
  def record_audio(duration, sr=16000):
      audio = sd.rec(int(duration * sr), samplerate=sr, channels=1)
      sd.wait()
      return audio.flatten()
  ```
- [ ] Silero VAD entegrasyonu
- [ ] Sessizlik algılama eşiğinin yapılandırması (10-15 sn)
- [ ] Otomatik kayıt başlatma/durdurma mantığı
- [ ] Kayıt-sonra-çevir (record-then-transcribe) akışı
- [ ] Farklı ortamlarda VAD testi

---

### Faz 4: Ses Ön İşleme Pipeline'ı

**Amaç:** WER iyileştirmesi için ses ön işleme modülünün geliştirilmesi

**Teknolojiler:**
- librosa (ses işleme)
- noisereduce (gürültü azaltma)
- scipy.signal (filtreleme)
- pydub (format dönüşümü)

**Kontrol Listesi:**
- [ ] 16kHz resampling modülü
- [ ] Mono dönüşüm
- [ ] Peak normalization
- [ ] Sessizlik kırpma (silence trimming)
- [ ] Spectral gating (opsiyonel)
- [ ] Low-pass filter (opsiyonel)
- [ ] Ön işleme ile/olmadan WER karşılaştırması
- [ ] Modüler pipeline tasarımı (açılıp kapanabilir adımlar)
  ```python
  class AudioPreprocessor:
      def __init__(self, 
                   resample=True,
                   normalize=True,
                   trim_silence=True,
                   denoise=False):
          self.steps = []
          if resample: self.steps.append(self._resample)
          if normalize: self.steps.append(self._normalize)
          # ...
  ```

---

### Faz 5: Dataset Hazırlığı ve Fine-tuning

**Amaç:** Whisper modelinin Türkçe için fine-tune edilmesi

**Teknolojiler:**
- HuggingFace Transformers
- HuggingFace Datasets
- PEFT (LoRA fine-tuning)
- Weights & Biases (izleme - opsiyonel)

**Kontrol Listesi:**
- [ ] Common Voice Türkçe dataset indirme
  ```python
  from datasets import load_dataset
  
  dataset = load_dataset(
      "mozilla-foundation/common_voice_17_0", 
      "tr",
      trust_remote_code=True
  )
  ```
- [ ] Dataset formatının Whisper için hazırlanması
- [ ] Train/validation/test bölümlemesi
- [ ] LoRA konfigürasyonu (parametre-verimli fine-tuning)
  ```python
  from peft import LoraConfig, get_peft_model
  
  lora_config = LoraConfig(
      r=32,
      lora_alpha=64,
      target_modules=["q_proj", "v_proj"],
      lora_dropout=0.05,
      bias="none"
  )
  ```
- [ ] Fine-tuning script'inin hazırlanması
- [ ] Eğitim sürecinin izlenmesi (loss, WER)
- [ ] Checkpoint kaydetme stratejisi
- [ ] Fine-tuned model değerlendirmesi
- [ ] WER hedefinin kontrolü (≤%8)

---

### Faz 6: Streamlit Arayüzü

**Amaç:** Kullanıcı dostu web arayüzünün geliştirilmesi

**Teknolojiler:**
- Streamlit
- streamlit-webrtc (mikrofon erişimi)
- st.session_state (durum yönetimi)

**Arayüz Bileşenleri:**
- Mikrofon kaydı başlatma/durdurma butonu
- Ses dosyası yükleme alanı
- Dil seçimi (Türkçe/İngilizce/Otomatik)
- Transkripsiyon sonuç alanı
- İndirme butonu (metin dosyası)
- VAD durum göstergesi
- İşlem süresi bilgisi

**Kontrol Listesi:**
- [ ] Temel Streamlit uygulaması iskeleti
  ```python
  import streamlit as st
  
  st.title("🎤 Konuşma Tanıma Sistemi")
  
  tab1, tab2 = st.tabs(["Mikrofon", "Dosya Yükle"])
  ```
- [ ] Dosya yükleme widget'ı
  ```python
  uploaded_file = st.file_uploader(
      "Ses dosyası yükleyin",
      type=["wav", "mp3", "m4a", "ogg"]
  )
  ```
- [ ] Mikrofon kaydı entegrasyonu
- [ ] Transkripsiyon tetikleme ve gösterim
- [ ] VAD görsel feedback
- [ ] Hata yönetimi ve kullanıcı bildirimleri
- [ ] Responsive tasarım kontrolü

---

### Faz 7: Test ve Değerlendirme

**Amaç:** Sistemin kapsamlı test edilmesi ve performans raporlaması

**Test Kategorileri:**
- WER hesaplama (jiwer kütüphanesi)
- Uzun/kısa kayıtlar
- Kod-değiştirme (code-switching) testleri

**Kontrol Listesi:**
- [ ] Test dataset hazırlığı
- [ ] WER ölçüm script'i
  ```python
  from jiwer import wer
  
  error_rate = wer(reference, hypothesis)
  print(f"WER: {error_rate:.2%}")
  ```
- [ ] Türkçe WER ölçümü
- [ ] İngilizce WER ölçümü
- [ ] İşlem süresi benchmarkları
- [ ] VAD doğruluk testi
- [ ] Kullanılabilirlik testi
- [ ] Hata analizi ve iyileştirme önerileri

---

### Faz 8: Dokümantasyon ve Rapor

**Amaç:** Akademik rapor ve kullanım dokümantasyonunun hazırlanması

**Rapor Bölümleri (IEEE Formatı):**
1. Abstract / Özet
2. Introduction / Giriş
3. Literature Review / Literatür Taraması
4. Methodology / Yöntem
5. Implementation / Uygulama
6. Experimental Results / Deneysel Sonuçlar
7. Discussion / Tartışma
8. Conclusion / Sonuç
9. References / Kaynaklar

**Kontrol Listesi:**
- [ ] Teknik rapor yazımı
- [ ] Model seçimi gerekçelendirmesi
- [ ] WER sonuçları tabloları ve grafikleri
- [ ] Karşılaştırmalı analiz
- [ ] Kod dokümantasyonu (README)
- [ ] Kullanım kılavuzu
- [ ] GitHub repository düzenlemesi

---

## 7. Teknoloji Yığını Özeti

| Kategori | Teknoloji | Kullanım Amacı |
|----------|-----------|----------------|
| **ASR Modeli** | Whisper Medium | Konuşmadan metne dönüşüm |
| **VAD** | Silero VAD | Sessizlik algılama |
| **Framework** | PyTorch + MPS | Apple Silicon optimizasyonu |
| **Fine-tuning** | HuggingFace + PEFT/LoRA | Parametre-verimli eğitim |
| **UI** | Streamlit | Web tabanlı arayüz |
| **Ses İşleme** | librosa, pydub, noisereduce | Ön işleme ve format dönüşümü |
| **Mikrofon** | sounddevice / PyAudio | Ses yakalama |
| **Değerlendirme** | jiwer | WER hesaplama |
| **Dataset** | Mozilla Common Voice TR | Türkçe fine-tuning verisi |

---

## 8. Kaynaklar

1. Radford, A., et al. "Robust Speech Recognition via Large-Scale Weak Supervision." *arXiv preprint arXiv:2212.04356*, 2022.

2. Mozilla Common Voice. https://commonvoice.mozilla.org

3. Silero VAD. https://github.com/snakers4/silero-vad

4. ISSAI Turkish Speech Corpus. https://huggingface.co/datasets/issai/Turkish_Speech_Corpus

5. "Implementation of a Whisper Architecture-Based Turkish ASR System and Evaluation of the Effect of Fine-Tuning with a Low-Rank Adaptation (LoRA) Adapter on Its Performance." *MDPI Electronics*, 2024.

6. Mercan, Ö.B., et al. "Performance Comparison of Pre-trained Models for Speech-to-Text in Turkish." *arXiv:2307.04765*, 2023.

7. Mussakhojayeva, S., et al. "Multilingual Speech Recognition for Turkic Languages." *ISSAI*, 2023.

8. HuggingFace Transformers. https://huggingface.co/docs/transformers

9. PEFT: Parameter-Efficient Fine-Tuning. https://github.com/huggingface/peft

10. Streamlit Documentation. https://docs.streamlit.io

---

*— Doküman Sonu —*
