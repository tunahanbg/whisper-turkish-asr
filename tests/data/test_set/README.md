# Test Veri Seti

ASR benchmark için hazırlanmış test veri seti.

## Genel Bilgiler

| Özellik           | Değer                              |
| ----------------- | ---------------------------------- |
| **Örnek Sayısı**  | 300                                |
| **Toplam Süre**   | ~72 dakika (4312.5 saniye)         |
| **Ortalama Süre** | ~14.4 saniye/örnek                 |
| **Dil**           | Türkçe                             |
| **Format**        | FLAC (16-bit)                      |
| **Sample Rate**   | 16kHz (orijinal kayıtlar değişken) |
| **Kaynak**        | Mozilla Common Voice Turkish       |

## Dizin Yapısı

```
test_set/
├── README.md              # Bu dosya
├── ground_truth.json      # 300 örnek için groundtruth transkriptler
└── audio/                 # Ses dosyaları
    ├── sample_001.flac
    ├── sample_002.flac
    ├── ...
    └── sample_300.flac
```

## Ground Truth Formatı

`ground_truth.json` dosyası şu yapıdadır:

```json
{
    "dataset_info": {
        "name": "ASR Test Set",
        "version": "1.0",
        "language": "tr",
        "sample_count": 300,
        "total_duration": 4312.5,
        "random_seed": 42
    },
    "test_samples": [
        {
            "id": "sample_001",
            "audio_file": "audio/sample_001.flac",
            "original_id": "uuid-here",
            "language": "tr",
            "duration": 14.38,
            "transcript": "ses kaydının tam transkripti buraya gelir",
            "metadata": {
                "source": "common_voice_tr",
                "format": "flac"
            }
        }
    ]
}
```

## Özellikler

### Veri Seçimi

-   **Rastgele Örnekleme:** Seed=42 ile reproducible seçim
-   **Kaynak:** `data/raw/TR` dizinindeki 2513 FLAC dosyasından
-   **Filtreleme:** Boş transkriptler ve hatalı dosyalar otomatik filtrelendi

### Groundtruth Kalitesi

-   ✅ **Manuel transkriptler** (Common Voice gönüllüleri tarafından doğrulanmış)
-   ✅ **Temiz metin** (noktalama işaretleri ve büyük/küçük harf korunmuş)
-   ✅ **Türkçe karakterler** (ç, ğ, ı, ö, ş, ü destekleniyor)

### Ses Kalitesi

-   **Format:** FLAC (kayıpsız sıkıştırma)
-   **Kalite:** Yüksek kalite (profesyonel olmayan kayıtlar)
-   **Konuşmacılar:** Farklı konuşmacılar (demografik çeşitlilik)
-   **Ortam:** Genelde sessiz ortamlar (ev kayıtları)

## Kullanım

### Python ile Yükleme

```python
import json
from pathlib import Path

# Ground truth'u yükle
with open('tests/data/test_set/ground_truth.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

test_samples = data['test_samples']
print(f"Loaded {len(test_samples)} samples")

# İlk örnek
sample = test_samples[0]
print(f"ID: {sample['id']}")
print(f"Audio: {sample['audio_file']}")
print(f"Duration: {sample['duration']}s")
print(f"Transcript: {sample['transcript']}")
```

### Ses Dosyası Yükleme

```python
import librosa

# Ses dosyasını yükle
audio_path = Path('tests/data/test_set') / sample['audio_file']
audio, sr = librosa.load(audio_path, sr=16000)

print(f"Audio shape: {audio.shape}")
print(f"Sample rate: {sr}Hz")
print(f"Duration: {librosa.get_duration(y=audio, sr=sr):.2f}s")
```

### Benchmark ile Kullanım

```python
from tests.evaluation.benchmarker import ASRBenchmarker

# Benchmarker oluştur
benchmarker = ASRBenchmarker(test_set_path='./tests/data/test_set')

# Test çalıştır
model_config = {
    'name': 'faster-whisper',
    'variant': 'base',
    'compute_type': 'int8',
}

result = benchmarker.run_single_test(model_config=model_config)
```

## İstatistikler

### Süre Dağılımı

```
Min:    ~3 saniye
Max:    ~25 saniye
Median: ~14 saniye
Mean:   14.38 saniye
```

### Kelime Sayısı (Tahmini)

-   Ortalama konuşma hızı: ~3 kelime/saniye
-   Toplam kelime: ~13,000 kelime
-   Ortalama kelime/örnek: ~43 kelime

## Test Seti Yeniden Oluşturma

Test setini yeniden oluşturmak için:

```bash
python tests/scripts/prepare_test_set.py
```

**Not:** Seed=42 kullanıldığı için aynı örnekler seçilir (reproducibility).

Farklı örnekler için config'de seed'i değiştirin:

```yaml
# config/config.yaml
evaluation:
    dataset_preparation:
        seed: 123 # Yeni seed
```

## Lisans

Bu test seti Mozilla Common Voice projesinden türetilmiştir:

-   **Lisans:** Creative Commons Zero (CC-0)
-   **Kaynak:** [Mozilla Common Voice](https://commonvoice.mozilla.org)
-   **Dil:** Turkish (tr)
-   **Versiyon:** Common Voice 17.0

## Sınırlamalar

1. **Doğal Değil:** Okunan metinler (spontane konuşma değil)
2. **Ses Kalitesi:** Değişken (profesyonel olmayan kayıtlar)
3. **Gürültü:** Minimum (gerçek dünya uygulamaları daha gürültülü)
4. **Domain:** Genel konuşma (teknik/medikal jargon yok)
5. **Konuşmacı Çeşitliliği:** İyi ama sınırlı (yaş, aksan, cinsiyet)

## İyileştirme Önerileri

Gelecekte test setini geliştirmek için:

-   [ ] Daha fazla örnek (500-1000)
-   [ ] Gürültülü ortam kayıtları ekle
-   [ ] Farklı domainler (haber, telefon, toplantı)
-   [ ] Kod-değiştirme örnekleri (Türkçe-İngilizce karışık)
-   [ ] Uzun konuşmalar (>30 saniye)
-   [ ] İngilizce örnekler (karşılaştırma için)

## Referanslar

1. Mozilla Common Voice: https://commonvoice.mozilla.org
2. Common Voice Paper: https://arxiv.org/abs/1912.06670
3. Türkçe ASR Literatürü: MDPI Electronics 2024

---

**Son Güncelleme:** Aralık 2025  
**Hazırlayan:** Tunahan - Gazi Üniversitesi
