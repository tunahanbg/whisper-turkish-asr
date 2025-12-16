#!/usr/bin/env python3
"""
Model karşılaştırma scripti.
Farklı model boyutlarını WER, hız ve kaynak kullanımı açısından karşılaştırır.
"""

import sys
from pathlib import Path
from loguru import logger

# Proje kök dizinini path'e ekle
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tests.evaluation.benchmarker import ASRBenchmarker
from tests.evaluation.report_generator import ReportGenerator


def print_comparison_table(results: list):
    """Karşılaştırma tablosunu yazdır."""
    print("\n" + "="*90)
    print("MODEL KARŞILAŞTIRMA TABLOSU")
    print("="*90)
    
    # Başlık
    print(f"{'Model':<20} {'Variant':<10} {'WER':<8} {'CER':<8} {'RTF':<8} {'CPU %':<8} {'Memory (MB)':<12}")
    print("-"*90)
    
    # Her model için satır
    for result in results:
        model_info = result['model']
        model_name = model_info['name']
        variant = model_info['variant']
        
        metrics = result['metrics']['turkish']
        wer = metrics['wer']
        cer = metrics['cer']
        rtf = metrics['avg_rtf']
        
        resources = result['resources']
        cpu = resources['avg_cpu_percent']
        memory = resources['peak_memory_mb']
        
        print(f"{model_name:<20} {variant:<10} {wer:>6.2%} {cer:>6.2%} {rtf:>7.2f} {cpu:>7.1f} {memory:>11.1f}")
    
    print("="*90)
    
    # En iyi modeller
    print("\nEN İYİ MODELLER:")
    print("-"*90)
    
    # En düşük WER
    best_wer = min(results, key=lambda x: x['metrics']['turkish']['wer'])
    print(f"✓ En Düşük WER: {best_wer['model']['name']} {best_wer['model']['variant']} "
          f"({best_wer['metrics']['turkish']['wer']:.2%})")
    
    # En hızlı (RTF)
    fastest = min(results, key=lambda x: x['metrics']['turkish']['avg_rtf'])
    print(f"✓ En Hızlı: {fastest['model']['name']} {fastest['model']['variant']} "
          f"(RTF: {fastest['metrics']['turkish']['avg_rtf']:.2f})")
    
    # En az memory kullanan
    least_memory = min(results, key=lambda x: x['resources']['peak_memory_mb'])
    print(f"✓ En Az Memory: {least_memory['model']['name']} {least_memory['model']['variant']} "
          f"({least_memory['resources']['peak_memory_mb']:.1f}MB)")
    
    print("="*90)


def main():
    """Ana fonksiyon."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Model Karşılaştırma")
    parser.add_argument(
        '--models',
        nargs='+',
        default=['tiny', 'base', 'small', 'medium'],
        help='Karşılaştırılacak model boyutları (tiny, base, small, medium, large)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Test edilecek örnek sayısı'
    )
    parser.add_argument(
        '--save',
        action='store_true',
        help='Sonuçları dosyaya kaydet'
    )
    
    args = parser.parse_args()
    
    logger.info("="*70)
    logger.info("MODEL KARŞILAŞTIRMA")
    logger.info("="*70)
    logger.info(f"Modeller: {', '.join(args.models)}")
    logger.info(f"Sample limit: {args.limit or 'All'}")
    logger.info("="*70 + "\n")
    
    # Custom models için config yükle
    from config import config as global_config
    custom_models_config = global_config.get('evaluation.custom_models', {})
    
    # Benchmarker oluştur
    test_set_path = "./tests/data/test_set"
    benchmarker = ASRBenchmarker(test_set_path=test_set_path)
    
    # Model karşılaştırması çalıştır
    try:
        results = benchmarker.run_model_comparison(
            models=args.models,
            implementation='faster-whisper',
            sample_limit=args.limit,
        )
        
        # Karşılaştırma tablosunu göster
        print_comparison_table(results)
        
        # Sonuçları kaydet (opsiyonel)
        if args.save:
            logger.info("\nSonuçlar kaydediliyor...")
            
            benchmark_results = {
                'benchmark_info': {
                    'mode': 'model_comparison',
                    'models': args.models,
                    'test_set_size': len(benchmarker.test_samples),
                    'sample_limit': args.limit,
                },
                'results': results,
            }
            
            reporter = ReportGenerator(results_dir='./tests/data/results')
            saved_files = reporter.save_results(
                benchmark_results,
                formats=['json', 'csv', 'markdown'],
                base_filename='model_comparison',
            )
            
            logger.success("\nSonuçlar kaydedildi:")
            for fmt, filepath in saved_files.items():
                logger.success(f"✓ {fmt.upper()}: {filepath}")
        
        logger.success("\n✓ Model karşılaştırma tamamlandı!")
    
    except Exception as e:
        logger.error(f"Model karşılaştırma başarısız: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
