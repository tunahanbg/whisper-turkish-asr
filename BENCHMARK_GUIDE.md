# ASR Benchmark Sistemi - Hızlı Başlangıç Kılavuzu

## 🎯 Sistem Durumu

✅ **Test altyapısı hazır!**
- 300 örneklik test veri seti oluşturuldu
- Kapsamlı benchmark framework kuruldu
- Metrik hesaplama ve raporlama sistemi hazır

## 🚀 Hızlı Başlangıç

### 1. Hızlı Test (5 örnek - ~1 dakika)

```bash
python tests/scripts/quick_test.py
```

Bu komut:
- faster-whisper base model ile 5 örnek test eder
- WER, CER, RTF metriklerini hesaplar
- Sisteminizin çalıştığını doğrular

### 2. Model Karşılaştırma (50-100 örnek - ~10-20 dakika)

```bash
python tests/scripts/compare_models.py --limit 50 --save
```

Bu komut:
- tiny, base, small modellerini karşılaştırır
- 50 örnekle hızlı karşılaştırma yapar
- Sonuçları dosyaya kaydeder

### 3. Tam Benchmark (300 örnek - ~1-1.5 saat)

```bash
python tests/scripts/run_benchmarks.py --mode full
```

Bu komut:
- Tüm model boyutlarını test eder
- whisper vs faster-whisper karşılaştırması yapar
- Preprocessing etkisini ölçer
- Kapsamlı raporlar oluşturur (JSON, CSV, Markdown)

## 📊 Test Modları

### Quick Test
```bash
python tests/scripts/run_benchmarks.py --mode quick
```
10 örnekle hızlı kontrol (~2 dakika)

### Sadece Model Karşılaştırma
```bash
python tests/scripts/run_benchmarks.py --mode models --limit 100
```
100 örnekle model boyutları (~20 dakika)

### Implementasyon Karşılaştırma
```bash
python tests/scripts/run_benchmarks.py --mode implementations
```
whisper vs faster-whisper (~30 dakika, 300 örnek)

### Preprocessing Testi
```bash
python tests/scripts/run_benchmarks.py --mode preprocessing
```
Preprocessing etkisini ölç (~30 dakika, 300 örnek)

## 📁 Sonuçlar

Sonuçlar `tests/data/results/` dizininde saklanır:

- `benchmark_*.json` - Detaylı sonuçlar
- `benchmark_*.csv` - Tablo formatı (Excel, pandas için)
- `benchmark_*.md` - İnsan okunabilir rapor
- `benchmark_*.tex` - LaTeX tablo (akademik rapor için)

## 📈 Beklenen Performans

Faster-whisper ile (MacBook Pro M4 Pro):

| Model | WER (Tahmini) | RTF | Memory |
|-------|---------------|-----|---------|
| tiny  | ~25%         | 0.15| ~280MB  |
| base  | ~18%         | 0.22| ~420MB  |
| small | ~15%         | 0.45| ~680MB  |
| medium| ~12%         | 0.80| ~1.2GB  |

**Not:** Bu tahminlerdir. Gerçek sonuçlar benchmark ile ölçülecektir.

## 🔧 Sorun Giderme

### "Ground truth file not found"

```bash
python tests/scripts/prepare_test_set.py
```

### Memory Hatası

Örnek sayısını azaltın:
```bash
python tests/scripts/run_benchmarks.py --mode full --limit 50
```

### Import Hatası

Proje kök dizininden çalıştırın:
```bash
cd /path/to/ASR_School_Project
python tests/scripts/quick_test.py
```

## 📚 Detaylı Dokümantasyon

- `tests/README.md` - Tam dokümantasyon
- `tests/data/test_set/README.md` - Test seti detayları

## 🎓 Faz 8 İçin

Akademik rapor için sonuçlar hazır:

```bash
# Tam benchmark + LaTeX export
python tests/scripts/run_benchmarks.py --mode full --formats json csv markdown latex
```

LaTeX tabloları `tests/data/results/*.tex` dosyalarında.

---

**İyi testler!** 🚀
