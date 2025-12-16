"""
Ana benchmark framework modülü.
Farklı modeller, implementasyonlar ve konfigürasyonları test eder.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from tqdm import tqdm
import librosa
from loguru import logger

from config import config
from src.models.model_manager import ModelManager
from src.preprocessing.processor import AudioPreprocessor
from .metrics import (
    calculate_wer,
    calculate_cer,
    calculate_rtf,
    detailed_error_analysis,
)
from .resource_monitor import ResourceMonitor, get_system_info


class ASRBenchmarker:
    """
    Modüler ASR benchmark framework.
    
    Testler:
    - Model boyutları (tiny, base, small, medium)
    - Implementasyonlar (whisper vs faster-whisper)
    - Preprocessing varyasyonları
    """
    
    def __init__(
        self,
        test_set_path: str,
        config_override: Optional[Dict] = None,
    ):
        """
        Args:
            test_set_path: Test veri seti dizini (ground_truth.json içerir)
            config_override: Config override (test için)
        """
        self.test_set_path = Path(test_set_path)
        self.config_override = config_override or {}
        
        # Test setini yükle
        self.test_samples = self._load_test_set()
        
        # Model manager
        self.model_manager = ModelManager()
        
        # Sonuçlar
        self.results = []
        
        logger.info(f"ASRBenchmarker initialized - {len(self.test_samples)} test samples")
    
    def _load_test_set(self) -> List[Dict[str, Any]]:
        """Test setini yükle (ground_truth.json)."""
        gt_file = self.test_set_path / "ground_truth.json"
        
        if not gt_file.exists():
            raise FileNotFoundError(f"Ground truth file not found: {gt_file}")
        
        with open(gt_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        test_samples = data.get('test_samples', [])
        logger.info(f"Loaded {len(test_samples)} test samples from {gt_file}")
        
        return test_samples
    
    def run_single_test(
        self,
        model_config: Dict[str, Any],
        preprocessing_config: Optional[Dict[str, Any]] = None,
        sample_limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Tek bir konfigürasyon için test çalıştır.
        
        Args:
            model_config: Model konfigürasyonu
                {
                    'name': 'faster-whisper',
                    'variant': 'base',
                    'compute_type': 'int8'
                }
            preprocessing_config: Preprocessing konfigürasyonu (None = disabled)
            sample_limit: Test edilecek maksimum örnek sayısı (None = hepsi)
        
        Returns:
            Test sonuçları dict
        """
        logger.info(f"Starting test: {model_config['name']} {model_config['variant']}")
        
        # Model yükle
        logger.info("Loading model...")
        
        # Config'i geçici olarak override et
        from config import config as global_config
        
        # Orijinal değerleri sakla
        orig_name = global_config.get('model.name')
        orig_variant = global_config.get('model.variant')
        orig_compute = global_config.get('model.compute_type')
        orig_model_path = global_config.get('model.model_path')
        orig_device = global_config.get('model.device')
        
        # Benchmark için override
        global_config.set('model.name', model_config['name'])
        global_config.set('model.variant', model_config['variant'])
        global_config.set('model.compute_type', model_config.get('compute_type', 'int8'))
        
        # Device (GPU/MPS support)
        if 'device' in model_config:
            global_config.set('model.device', model_config['device'])
        
        # Custom model path (quantized/fine-tuned models için)
        if 'model_path' in model_config:
            global_config.set('model.model_path', model_config['model_path'])
        
        try:
            # Model oluştur ve yükle
            from src.models.faster_whisper_model import FasterWhisperASR
            from src.models.whisper_model import WhisperASR
            
            if model_config['name'] == 'faster-whisper':
                model = FasterWhisperASR()
            else:
                model = WhisperASR()
            
            model.load()
            
            # ModelManager'a kaydet
            self.model_manager.current_model = model
            self.model_manager.model_type = model_config['name']
        
        finally:
            # Config'i geri yükle
            if orig_name:
                global_config.set('model.name', orig_name)
            if orig_variant:
                global_config.set('model.variant', orig_variant)
            if orig_compute:
                global_config.set('model.compute_type', orig_compute)
            # Restore model_path (None için de restore et)
            global_config.set('model.model_path', orig_model_path)
            # Restore device
            if orig_device:
                global_config.set('model.device', orig_device)
        
        # Preprocessor
        preprocessor = None
        if preprocessing_config and preprocessing_config.get('enabled', False):
            preprocessor = AudioPreprocessor(custom_config=preprocessing_config)
            logger.info(f"Preprocessing enabled: {preprocessor}")
        else:
            logger.info("Preprocessing disabled")
        
        # Test örneklerini seç
        test_samples = self.test_samples
        if sample_limit:
            test_samples = test_samples[:sample_limit]
        
        # Her örnek için test çalıştır
        sample_results = []
        total_wer = 0.0
        total_cer = 0.0
        total_rtf = 0.0
        
        for sample in tqdm(test_samples, desc=f"Testing {model_config['variant']}"):
            sample_id = sample['id']
            audio_path = self.test_set_path / sample['audio_file']
            reference_text = sample['transcript']
            
            try:
                # Ses dosyasını yükle
                audio, sr = librosa.load(audio_path, sr=None)
                audio_duration = librosa.get_duration(y=audio, sr=sr)
                
                # Preprocessing (opsiyonel)
                if preprocessor:
                    audio = preprocessor.process(audio, sr)
                
                # Transkripsiyon (kaynak izlemeli)
                with ResourceMonitor() as monitor:
                    start_time = time.time()
                    
                    # Model transcribe (numpy array gönder, tüm modeller için uyumlu)
                    result = model.transcribe(audio)
                    hypothesis_text = result['text'].strip()
                    
                    process_time = time.time() - start_time
                
                resource_stats = monitor.get_stats()
                
                # Metrikleri hesapla
                # 1. Raw metrikler (orijinal metinler)
                wer_raw = calculate_wer(reference_text, hypothesis_text)
                cer_raw = calculate_cer(reference_text, hypothesis_text)
                
                # 2. Normalized metrikler (Türkçe için daha adil)
                from .metrics import normalize_turkish_text, calculate_wer_normalized
                ref_normalized = normalize_turkish_text(reference_text)
                hyp_normalized = normalize_turkish_text(hypothesis_text)
                wer_normalized = calculate_wer(ref_normalized, hyp_normalized)
                cer_normalized = calculate_cer(ref_normalized, hyp_normalized)
                
                # 3. Diğer metrikler
                rtf = calculate_rtf(audio_duration, process_time)
                error_analysis = detailed_error_analysis(ref_normalized, hyp_normalized)
                
                # Örnek sonucu kaydet
                sample_result = {
                    'sample_id': sample_id,
                    'audio_duration': audio_duration,
                    'reference': reference_text,
                    'hypothesis': hypothesis_text,
                    'reference_normalized': ref_normalized,
                    'hypothesis_normalized': hyp_normalized,
                    # Normalized metrikler (varsayılan)
                    'wer': wer_normalized,
                    'cer': cer_normalized,
                    # Raw metrikler (karşılaştırma için)
                    'wer_raw': wer_raw,
                    'cer_raw': cer_raw,
                    # Diğer metrikler
                    'rtf': rtf,
                    'process_time': process_time,
                    'error_analysis': error_analysis,
                    'resources': {
                        'cpu_percent': resource_stats.cpu_percent_avg,
                        'memory_mb': resource_stats.memory_mb_peak,
                        'memory_increase_mb': resource_stats.memory_mb_increase,
                    }
                }
                
                sample_results.append(sample_result)
                
                # Toplam metrikleri güncelle (normalized kullan)
                total_wer += wer_normalized
                total_cer += cer_normalized
                total_rtf += rtf
                
                logger.debug(f"Sample {sample_id}: WER={wer_normalized:.2%} (raw: {wer_raw:.2%}), "
                           f"CER={cer_normalized:.2%}, RTF={rtf:.2f}")
            
            except Exception as e:
                logger.error(f"Failed to process sample {sample_id}: {e}")
                # Hatalı örneği kaydet
                sample_results.append({
                    'sample_id': sample_id,
                    'error': str(e),
                    'wer': 1.0,
                    'cer': 1.0,
                })
        
        # Ortalama metrikleri hesapla
        n_samples = len(sample_results)
        avg_wer = total_wer / n_samples if n_samples > 0 else 1.0
        avg_cer = total_cer / n_samples if n_samples > 0 else 1.0
        avg_rtf = total_rtf / n_samples if n_samples > 0 else 0.0
        
        # Toplam kaynak kullanımı
        avg_cpu = sum(r.get('resources', {}).get('cpu_percent', 0) 
                     for r in sample_results if 'resources' in r) / n_samples
        peak_memory = max(r.get('resources', {}).get('memory_mb', 0) 
                         for r in sample_results if 'resources' in r)
        
        # Model'i bellekten kaldır
        self.model_manager.unload_model()
        
        # Sonuç
        test_result = {
            'model': model_config,
            'preprocessing': preprocessing_config or {'enabled': False},
            'test_set_size': n_samples,
            'timestamp': datetime.now().isoformat(),
            'metrics': {
                'turkish': {  # Türkçe veri seti için
                    'wer': avg_wer,
                    'cer': avg_cer,
                    'avg_rtf': avg_rtf,
                }
            },
            'resources': {
                'avg_cpu_percent': avg_cpu,
                'peak_memory_mb': peak_memory,
            },
            'per_sample_results': sample_results,
        }
        
        logger.success(f"Test completed: WER={avg_wer:.2%}, CER={avg_cer:.2%}, RTF={avg_rtf:.2f}")
        
        return test_result
    
    def run_model_comparison(
        self,
        models: Optional[List[str]] = None,
        implementation: str = 'faster-whisper',
        sample_limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Farklı model boyutlarını karşılaştır.
        
        Args:
            models: Test edilecek model boyutları ['tiny', 'base', 'small', 'medium', 'large-v3-w4a16']
            implementation: Implementasyon ('whisper' veya 'faster-whisper')
            sample_limit: Örnek sayısı limiti
        
        Returns:
            Tüm test sonuçları
        """
        if models is None:
            models = config.get('evaluation.model_variants', ['tiny', 'base', 'small'])
        
        logger.info(f"Model comparison: {models} ({implementation})")
        
        # Custom models config'ini yükle
        custom_models_config = config.get('evaluation.custom_models', {})
        
        results = []
        
        for variant in models:
            # Check if this is a custom model
            is_custom = False
            model_config = None
            
            # Custom model variant check (e.g., "large-v3-w4a16")
            for custom_key, custom_cfg in custom_models_config.items():
                if custom_cfg.get('variant') == variant:
                    is_custom = True
                    model_config = {
                        'name': custom_cfg.get('name', 'whisper'),
                        'variant': variant,
                        'compute_type': custom_cfg.get('compute_type', 'float16'),
                        'model_path': custom_cfg.get('model_path'),
                        'device': 'cpu',  # Force CPU for quantized models (stability)
                    }
                    logger.info(f"Using custom model config for {variant} (device: cpu, forced for stability)")
                    break
            
            # Standard model
            if not is_custom:
                model_config = {
                    'name': implementation,
                    'variant': variant,
                    'compute_type': 'int8',
                }
            
            result = self.run_single_test(
                model_config=model_config,
                preprocessing_config=None,
                sample_limit=sample_limit,
            )
            
            results.append(result)
        
        return results
    
    def run_implementation_comparison(
        self,
        variant: str = 'base',
        sample_limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        whisper vs faster-whisper karşılaştırması.
        
        Args:
            variant: Model boyutu
            sample_limit: Örnek sayısı limiti
        
        Returns:
            Karşılaştırma sonuçları
        """
        logger.info(f"Implementation comparison: whisper vs faster-whisper ({variant})")
        
        results = []
        
        for impl in ['whisper', 'faster-whisper']:
            model_config = {
                'name': impl,
                'variant': variant,
                'compute_type': 'int8' if impl == 'faster-whisper' else None,
            }
            
            result = self.run_single_test(
                model_config=model_config,
                preprocessing_config=None,
                sample_limit=sample_limit,
            )
            
            results.append(result)
        
        return results
    
    def run_preprocessing_comparison(
        self,
        model_config: Optional[Dict[str, Any]] = None,
        sample_limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Preprocessing varyasyonlarını test et.
        
        Args:
            model_config: Model konfigürasyonu (None = base faster-whisper)
            sample_limit: Örnek sayısı limiti
        
        Returns:
            Preprocessing karşılaştırma sonuçları
        """
        if model_config is None:
            model_config = {
                'name': 'faster-whisper',
                'variant': 'base',
                'compute_type': 'int8',
            }
        
        logger.info("Preprocessing comparison")
        
        # Preprocessing konfigürasyonları
        preprocessing_configs = [
            {
                'name': 'no_preprocessing',
                'enabled': False,
            },
            {
                'name': 'basic',
                'enabled': True,
                'normalize': True,
                'trim_silence': True,
                'denoise': False,
            },
            {
                'name': 'full',
                'enabled': True,
                'normalize': True,
                'trim_silence': True,
                'denoise': True,
                'noise_reduction_strength': 0.8,
            },
        ]
        
        results = []
        
        for prep_config in preprocessing_configs:
            logger.info(f"Testing preprocessing: {prep_config['name']}")
            
            result = self.run_single_test(
                model_config=model_config,
                preprocessing_config=prep_config,
                sample_limit=sample_limit,
            )
            
            results.append(result)
        
        return results
    
    def run_full_benchmark(
        self,
        sample_limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Tüm testleri çalıştır (kapsamlı benchmark).
        
        Args:
            sample_limit: Örnek sayısı limiti
        
        Returns:
            Tüm sonuçlar içeren dict
        """
        logger.info("="*70)
        logger.info("FULL BENCHMARK BAŞLIYOR")
        logger.info("="*70)
        
        start_time = time.time()
        
        # Sistem bilgileri
        system_info = get_system_info()
        
        all_results = []
        
        # 1. Model boyutları karşılaştırması
        logger.info("\n[1/3] Model Boyutları Karşılaştırması...")
        model_results = self.run_model_comparison(
            models=['tiny', 'base', 'small'],
            implementation='faster-whisper',
            sample_limit=sample_limit,
        )
        all_results.extend(model_results)
        
        # 2. Implementation karşılaştırması (base model)
        logger.info("\n[2/3] Implementation Karşılaştırması...")
        impl_results = self.run_implementation_comparison(
            variant='base',
            sample_limit=sample_limit,
        )
        all_results.extend(impl_results)
        
        # 3. Preprocessing karşılaştırması
        logger.info("\n[3/3] Preprocessing Karşılaştırması...")
        prep_results = self.run_preprocessing_comparison(
            sample_limit=sample_limit,
        )
        all_results.extend(prep_results)
        
        # Toplam süre
        total_duration = time.time() - start_time
        duration_str = f"{int(total_duration // 60)}m {int(total_duration % 60)}s"
        
        # Nihai sonuç
        benchmark_results = {
            'benchmark_info': {
                'timestamp': datetime.now().isoformat(),
                'test_set_size': len(self.test_samples),
                'total_duration': duration_str,
                'sample_limit': sample_limit,
            },
            'system_info': system_info,
            'results': all_results,
        }
        
        logger.success("="*70)
        logger.success(f"BENCHMARK TAMAMLANDI - Süre: {duration_str}")
        logger.success("="*70)
        
        return benchmark_results
