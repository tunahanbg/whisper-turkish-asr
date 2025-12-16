#!/usr/bin/env python3
"""
Hızlı test scripti.
Tek bir model ile 5-10 örnek üzerinde hızlı test yapar.
Sistem kontrolü ve debug için kullanılır.
"""

import sys
from pathlib import Path
from loguru import logger

# Proje kök dizinini path'e ekle
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tests.evaluation.benchmarker import ASRBenchmarker
from tests.evaluation.report_generator import ReportGenerator


def main():
    """Ana fonksiyon."""
    logger.info("="*70)
    logger.info("HIZLI TEST - 5 ÖRNEK")
    logger.info("="*70)
    
    # Test parametreleri
    test_set_path = "./tests/data/test_set"
    sample_limit = 5
    
    # Model konfigürasyonu (en hızlı: faster-whisper base)
    model_config = {
        'name': 'faster-whisper',
        'variant': 'base',
        'compute_type': 'int8',
    }
    
    logger.info(f"Model: {model_config['name']} {model_config['variant']}")
    logger.info(f"Sample limit: {sample_limit}")
    logger.info("="*70 + "\n")
    
    # Benchmarker oluştur ve test çalıştır
    try:
        benchmarker = ASRBenchmarker(test_set_path=test_set_path)
        
        result = benchmarker.run_single_test(
            model_config=model_config,
            preprocessing_config=None,
            sample_limit=sample_limit,
        )
        
        # Sonuçları göster
        metrics = result['metrics']['turkish']
        
        print("\n" + "="*70)
        print("SONUÇLAR (Normalized Text)")
        print("="*70)
        print(f"WER: {metrics['wer']:.2%}")
        print(f"CER: {metrics['cer']:.2%}")
        print(f"RTF: {metrics['avg_rtf']:.2f}")
        
        resources = result['resources']
        print(f"CPU: {resources['avg_cpu_percent']:.1f}%")
        print(f"Memory: {resources['peak_memory_mb']:.1f}MB")
        print("="*70)
        
        # Raw metrikleri de göster (varsa)
        if 'wer_raw' in result['per_sample_results'][0]:
            raw_wer = sum(s.get('wer_raw', 0) for s in result['per_sample_results'] 
                         if 'error' not in s) / len(result['per_sample_results'])
            print(f"\nRAW WER (normalizasyon olmadan): {raw_wer:.2%}")
            print(f"İyileşme: {(raw_wer - metrics['wer']) / raw_wer * 100:.1f}%")
            print("="*70)
        
        # Örnek bazlı sonuçlar
        print("\nÖRNEK BAZLI SONUÇLAR:")
        print("-"*70)
        for sample_result in result['per_sample_results'][:5]:
            if 'error' in sample_result:
                print(f"{sample_result['sample_id']}: ERROR - {sample_result['error']}")
            else:
                wer_info = f"WER={sample_result['wer']:.2%}"
                if 'wer_raw' in sample_result:
                    wer_info += f" (raw: {sample_result['wer_raw']:.2%})"
                print(f"{sample_result['sample_id']}: {wer_info}, RTF={sample_result['rtf']:.2f}")
        print("="*70)
        
        logger.success("\n✓ Hızlı test başarıyla tamamlandı!")
        logger.info("Not: Tam benchmark için 'run_benchmarks.py --mode full' kullanın")
    
    except Exception as e:
        logger.error(f"Hızlı test başarısız: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
