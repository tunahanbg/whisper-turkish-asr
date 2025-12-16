# Test ve Değerlendirme Sistemi

Faz 7 kapsamında oluşturulan kapsamlı ASR test ve benchmark altyapısı.

## İçindekiler

-   [Genel Bakış](#genel-bakış)
-   [Dizin Yapısı](#dizin-yapısı)
-   [Kurulum](#kurulum)
-   [Kullanım](#kullanım)
-   [Metrikler](#metrikler)
-   [Sonuçlar](#sonuçlar)

## Genel Bakış

Bu test sistemi aşağıdaki özellikleri sağlar:

-   ✅ **300 örneklik test veri seti** (groundtruth'lu)
-   ✅ **Modüler benchmark framework** (model, implementasyon, preprocessing karşılaştırması)
-   ✅ **Kapsamlı metrikler** (WER, CER, RTF, CPU, memory)
-   ✅ **Otomatik raporlama** (JSON, CSV, Markdown, LaTeX)
-   ✅ **Faz 8 uyumlu** (akademik rapor için hazır)

## Dizin Yapısı

```
tests/
├── __init__.py
├── README.md                      # Bu dosya
├── evaluation/                    # Değerlendirme modülleri
│   ├── __init__.py
│   ├── metrics.py                 # WER, CER, RTF hesaplamaları
│   ├── benchmarker.py             # Ana benchmark framework
│   ├── resource_monitor.py        # CPU, memory monitoring
│   └── report_generator.py        # Sonuç raporlama
├── data/
│   ├── test_set/                  # Test veri seti
│   │   ├── audio/                 # 300 FLAC dosyası
│   │   ├── ground_truth.json      # Groundtruth transkriptler
│   │   └── README.md              # Test seti dok¨mantasyonu
│   └── results/                   # Benchmark sonuçları
│       ├── benchmark_*.json
│       ├── benchmark_*.csv
│       └── benchmark_*.md
└── scripts/                       # Test scriptleri
    ├── prepare_test_set.py        # Test seti hazırlama
    ├── run_benchmarks.py          # Ana benchmark runner
    ├── quick_test.py              # Hızlı test (5 örnek)
    └── compare_models.py          # Model karşılaştırma
```

## Kurulum

### 1. Bağımlılıkları Yükle

```bash
pip install -r requirements.txt
```

Önemli paketler:

-   `jiwer` - WER/CER hesaplama
-   `psutil` - Kaynak kullanımı izleme
-   `tqdm` - Progress bar
-   `pandas` - Data analizi ve export
-   `librosa` - Ses işleme

### 2. Test Veri Setini Hazırla

```bash
python tests/scripts/prepare_test_set.py
```

Bu script:

-   `data/raw/TR` dizininden 300 rastgele örnek seçer
-   FLAC dosyalarını `tests/data/test_set/audio/` dizinine kopyalar
-   `ground_truth.json` dosyasını oluşturur

**Not:** Test veri seti zaten hazırsa bu adımı atlayabilirsiniz.

## Kullanım

### Hızlı Test (5 örnek)

Sistem kontrolü ve debug için:

```bash
python tests/scripts/quick_test.py
```

**Çıktı örneği:**

```
WER: 12.45%
CER: 5.23%
RTF: 0.42
CPU: 67.3%
Memory: 512.1MB
```

### Model Karşılaştırma

Farklı model boyutlarını karşılaştır:

```bash
# Varsayılan (tiny, base, small)
python tests/scripts/compare_models.py

# Özel modeller ve örnek sayısı
python tests/scripts/compare_models.py --models tiny base small medium --limit 100

# Sonuçları kaydet
python tests/scripts/compare_models.py --save
```

### Ana Benchmark Runner

Modüler test sistemi - farklı modlar:

#### 1. Tam Benchmark (Tüm Testler)

```bash
python tests/scripts/run_benchmarks.py --mode full
```

Çalıştırılan testler:

-   Model boyutları (tiny, base, small)
-   Implementasyonlar (whisper vs faster-whisper)
-   Preprocessing varyasyonları (disabled, basic, full)

**Tahmini süre:** ~1-1.5 saat (300 örnek için)

#### 2. Sadece Model Karşılaştırma

```bash
python tests/scripts/run_benchmarks.py --mode models
```

#### 3. Implementasyon Karşılaştırması

```bash
python tests/scripts/run_benchmarks.py --mode implementations
```

Whisper vs faster-whisper karşılaştırması (base model).

#### 4. Preprocessing Testi

```bash
python tests/scripts/run_benchmarks.py --mode preprocessing
```

Preprocessing'in WER'e etkisini test eder.

#### 5. Hızlı Test

```bash
python tests/scripts/run_benchmarks.py --mode quick
```

10 örnekle hızlı kontrol.

### Ek Parametreler

```bash
# Örnek sayısını sınırla
python tests/scripts/run_benchmarks.py --mode full --limit 50

# Çıktı formatları
python tests/scripts/run_benchmarks.py --mode models --formats json csv markdown latex

# Özel test seti
python tests/scripts/run_benchmarks.py --mode full --test-set ./custom_test_set
```

## Metrikler

### WER (Word Error Rate)

```
WER = (Substitutions + Deletions + Insertions) / Total Words
```

-   **Düşük = İyi** (0% = mükemmel)
-   Kelime bazlı hata oranı
-   **Hedef:** ≤ %8 (Türkçe, fine-tuned model)

### CER (Character Error Rate)

```
CER = (Substitutions + Deletions + Insertions) / Total Characters
```

-   Karakter bazlı hata oranı
-   WER'den genelde daha düşük

### RTF (Real-Time Factor)

```
RTF = Processing Time / Audio Duration
```

-   **< 1.0:** Gerçek zamanlı işleme mümkün
-   **0.5:** 2x daha hızlı (10s ses → 5s işlem)
-   **2.0:** 2x daha yavaş

### Kaynak Kullanımı

-   **CPU Usage:** Ortalama CPU yüzdesi
-   **Peak Memory:** Maksimum memory kullanımı (MB)
-   **Memory Increase:** İşlem sırasında artış

## Sonuçlar

### Sonuç Formatları

Benchmark sonuçları birden fazla formatta kaydedilir:

#### 1. JSON (Detaylı)

```json
{
  "benchmark_info": {
    "timestamp": "2025-12-15T10:30:00",
    "test_set_size": 300,
    "total_duration": "1h 15m"
  },
  "system_info": {...},
  "results": [
    {
      "model": {...},
      "metrics": {...},
      "resources": {...},
      "per_sample_results": [...]
    }
  ]
}
```

**Kullanım:** Programatik analiz, grafikler

#### 2. CSV (Tablo)

```csv
model,variant,wer,cer,rtf,cpu_percent,memory_mb
faster-whisper,tiny,0.25,0.10,0.15,42.1,280.5
faster-whisper,base,0.18,0.07,0.22,58.3,420.1
...
```

**Kullanım:** Excel, pandas, istatistiksel analiz

#### 3. Markdown (İnsan Okunabilir)

Karşılaştırma tabloları, en iyi sonuçlar, özet bilgiler.

**Kullanım:** GitHub README, raporlar

#### 4. LaTeX (Akademik)

Faz 8 akademik raporu için hazır tablo.

**Kullanım:** IEEE formatı makaleler

### Sonuç Dizini

```
tests/data/results/
├── benchmark_20251215_103045.json
├── benchmark_20251215_103045.csv
├── benchmark_20251215_103045.md
├── benchmark_20251215_103045.tex
└── model_comparison.json
```

### Örnek Sonuç Özeti

```
=================================================================
ASR BENCHMARK SONUÇLARI
=================================================================

Model                Variant    WER      RTF     Memory
-----------------------------------------------------------------
faster-whisper       tiny       25.30%   0.15    280.5MB
faster-whisper       base       18.25%   0.22    420.1MB
faster-whisper       small      14.80%   0.45    680.3MB
whisper              base       18.50%   0.62    450.2MB
=================================================================

EN İYİ SONUÇLAR:
✓ En Düşük WER: faster-whisper small (14.80%)
✓ En Hızlı: faster-whisper tiny (RTF: 0.15)
```

## Faz 8 Entegrasyonu

Test sonuçları Faz 8 (Dokümantasyon ve Rapor) için hazır formattadır:

### Akademik Rapor için

-   **LaTeX tablolar:** `*.tex` dosyaları
-   **Grafik dataları:** CSV/JSON formatında
-   **İstatistiksel analiz:** Mean, std, confidence intervals

### Kullanım

```python
from tests.evaluation.report_generator import generate_comparison_table

# Markdown tablo
table_md = generate_comparison_table(results, format='markdown')

# LaTeX tablo
table_tex = generate_comparison_table(results, format='latex')
```

## İleri Seviye Kullanım

### Python API

```python
from tests.evaluation.benchmarker import ASRBenchmarker
from tests.evaluation.report_generator import ReportGenerator

# Benchmarker oluştur
benchmarker = ASRBenchmarker(test_set_path="./tests/data/test_set")

# Özel test
model_config = {
    'name': 'faster-whisper',
    'variant': 'medium',
    'compute_type': 'int8',
}

result = benchmarker.run_single_test(
    model_config=model_config,
    sample_limit=50,
)

# Sonuçları kaydet
reporter = ReportGenerator()
reporter.save_results({'results': [result]}, formats=['json', 'markdown'])
reporter.print_summary({'results': [result]})
```

### Özel Preprocessing

```python
preprocessing_config = {
    'enabled': True,
    'normalize': True,
    'trim_silence': True,
    'denoise': True,
    'noise_reduction_strength': 0.6,
}

result = benchmarker.run_single_test(
    model_config=model_config,
    preprocessing_config=preprocessing_config,
)
```

## Troubleshooting

### Hata: "Ground truth file not found"

```bash
# Test setini hazırla
python tests/scripts/prepare_test_set.py
```

### Hata: "ModuleNotFoundError: No module named 'config'"

Python path sorunu. Script'leri proje kök dizininden çalıştırın:

```bash
cd /path/to/ASR_School_Project
python tests/scripts/run_benchmarks.py --mode quick
```

### Memory Hatası

Örnek sayısını azaltın:

```bash
python tests/scripts/run_benchmarks.py --mode full --limit 50
```

## Gelecek Geliştirmeler

-   [ ] Multi-GPU desteği
-   [ ] Distributed benchmark (paralel test)
-   [ ] Web UI (benchmark sonuçlarını görselleştirme)
-   [ ] A/B testing framework
-   [ ] Otomatik model selection (en iyi model önerisi)

## İletişim

Sorular veya öneriler için:

-   **Geliştirici:** Tunahan
-   **Kurum:** Gazi Üniversitesi Bilgisayar Mühendisliği
-   **Proje:** ASR School Project - Faz 7

---

**Son Güncelleme:** Aralık 2025
