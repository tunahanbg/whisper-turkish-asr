#!/usr/bin/env python3
"""
Ana benchmark runner.
Modüler test sistemi - farklı test modları destekler.

Kullanım:
    python run_benchmarks.py --mode full          # Tüm testler
    python run_benchmarks.py --mode models        # Sadece model karşılaştırma
    python run_benchmarks.py --mode implementations # Whisper vs faster-whisper
    python run_benchmarks.py --mode preprocessing # Preprocessing testi
    python run_benchmarks.py --mode quick         # Hızlı test (10 örnek)
"""

import sys
import argparse
from pathlib import Path
from loguru import logger

# Proje kök dizinini path'e ekle
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tests.evaluation.benchmarker import ASRBenchmarker
from tests.evaluation.report_generator import ReportGenerator


def main():
    """Ana fonksiyon."""
    parser = argparse.ArgumentParser(
        description="ASR Benchmark Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Test Modları:
  full            Tüm testleri çalıştır (en kapsamlı)
  models          Model boyutları karşılaştır (tiny, base, small)
  implementations Whisper vs faster-whisper karşılaştır
  preprocessing   Preprocessing etkisini test et
  quick           Hızlı test (10 örnek ile)
  
Örnekler:
  python run_benchmarks.py --mode full
  python run_benchmarks.py --mode models --limit 50
  python run_benchmarks.py --mode quick
        """
    )
    
    parser.add_argument(
        '--mode',
        choices=['full', 'models', 'implementations', 'preprocessing', 'quick'],
        required=True,
        help='Test modu'
    )
    
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Test edilecek maksimum örnek sayısı (None = hepsi)'
    )
    
    parser.add_argument(
        '--test-set',
        type=str,
        default='./tests/data/test_set',
        help='Test veri seti dizini'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='./tests/data/results',
        help='Sonuç dizini'
    )
    
    parser.add_argument(
        '--formats',
        nargs='+',
        default=['json', 'csv', 'markdown'],
        choices=['json', 'csv', 'markdown', 'latex'],
        help='Çıktı formatları'
    )
    
    args = parser.parse_args()
    
    # Başlangıç logları
    logger.info("="*70)
    logger.info("ASR BENCHMARK RUNNER")
    logger.info("="*70)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Test set: {args.test_set}")
    logger.info(f"Sample limit: {args.limit or 'All'}")
    logger.info(f"Output formats: {', '.join(args.formats)}")
    logger.info("="*70)
    
    # Benchmarker oluştur
    benchmarker = ASRBenchmarker(test_set_path=args.test_set)
    
    # Test çalıştır
    results = None
    
    if args.mode == 'full':
        logger.info("Running FULL BENCHMARK...")
        results = benchmarker.run_full_benchmark(sample_limit=args.limit)
    
    elif args.mode == 'models':
        logger.info("Running MODEL COMPARISON...")
        models = ['tiny', 'base', 'small']
        model_results = benchmarker.run_model_comparison(
            models=models,
            implementation='faster-whisper',
            sample_limit=args.limit,
        )
        
        results = {
            'benchmark_info': {
                'mode': 'model_comparison',
                'test_set_size': len(benchmarker.test_samples),
                'sample_limit': args.limit,
            },
            'results': model_results,
        }
    
    elif args.mode == 'implementations':
        logger.info("Running IMPLEMENTATION COMPARISON...")
        impl_results = benchmarker.run_implementation_comparison(
            variant='base',
            sample_limit=args.limit,
        )
        
        results = {
            'benchmark_info': {
                'mode': 'implementation_comparison',
                'test_set_size': len(benchmarker.test_samples),
                'sample_limit': args.limit,
            },
            'results': impl_results,
        }
    
    elif args.mode == 'preprocessing':
        logger.info("Running PREPROCESSING COMPARISON...")
        prep_results = benchmarker.run_preprocessing_comparison(
            sample_limit=args.limit,
        )
        
        results = {
            'benchmark_info': {
                'mode': 'preprocessing_comparison',
                'test_set_size': len(benchmarker.test_samples),
                'sample_limit': args.limit,
            },
            'results': prep_results,
        }
    
    elif args.mode == 'quick':
        logger.info("Running QUICK TEST (10 samples)...")
        quick_limit = 10
        
        # Sadece base model ile hızlı test
        model_config = {
            'name': 'faster-whisper',
            'variant': 'base',
            'compute_type': 'int8',
        }
        
        test_result = benchmarker.run_single_test(
            model_config=model_config,
            preprocessing_config=None,
            sample_limit=quick_limit,
        )
        
        results = {
            'benchmark_info': {
                'mode': 'quick_test',
                'test_set_size': len(benchmarker.test_samples),
                'sample_limit': quick_limit,
            },
            'results': [test_result],
        }
    
    # Sonuçları kaydet
    if results:
        logger.info("\nSaving results...")
        reporter = ReportGenerator(results_dir=args.output)
        
        saved_files = reporter.save_results(results, formats=args.formats)
        
        # Özet yazdır
        reporter.print_summary(results)
        
        # Kaydedilen dosyaları göster
        logger.success("\n" + "="*70)
        logger.success("SONUÇLAR KAYDEDİLDİ")
        logger.success("="*70)
        for fmt, filepath in saved_files.items():
            logger.success(f"✓ {fmt.upper()}: {filepath}")
        logger.success("="*70)
    
    logger.info("\nBenchmark completed successfully!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("\nBenchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\nBenchmark failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
