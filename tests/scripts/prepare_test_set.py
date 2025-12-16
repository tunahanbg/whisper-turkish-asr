#!/usr/bin/env python3
"""
Test veri seti hazırlama scripti.
data/raw/TR dizininden rastgele 300 örnek seçer ve test seti oluşturur.
"""

import sys
import json
import random
import shutil
from pathlib import Path
from typing import List, Dict
import librosa
import yaml
from tqdm import tqdm
from loguru import logger

# Proje kök dizinini path'e ekle
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_config():
    """Config dosyasını yükle."""
    config_file = PROJECT_ROOT / "config" / "config.yaml"
    with open(config_file, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


config = load_config()


def find_audio_files(source_dir: Path) -> List[Path]:
    """
    Kaynak dizindeki .flac dosyalarını bul.
    
    Args:
        source_dir: Kaynak dizin
    
    Returns:
        .flac dosya yolları listesi
    """
    flac_files = list(source_dir.glob("*.flac"))
    logger.info(f"Found {len(flac_files)} FLAC files in {source_dir}")
    return flac_files


def select_random_samples(
    audio_files: List[Path],
    count: int,
    seed: int = 42,
) -> List[Path]:
    """
    Rastgele örnekler seç.
    
    Args:
        audio_files: Tüm ses dosyaları
        count: Seçilecek örnek sayısı
        seed: Random seed (reproducibility)
    
    Returns:
        Seçilen dosya yolları
    """
    random.seed(seed)
    
    if count > len(audio_files):
        logger.warning(f"Requested {count} samples but only {len(audio_files)} available. "
                      f"Using all samples.")
        count = len(audio_files)
    
    selected = random.sample(audio_files, count)
    logger.info(f"Selected {len(selected)} random samples (seed={seed})")
    
    return selected


def prepare_test_set(
    source_dir: str,
    test_set_dir: str,
    sample_count: int = 300,
    seed: int = 42,
) -> Dict:
    """
    Test veri setini hazırla.
    
    1. Rastgele örnekler seç
    2. FLAC dosyalarını kopyala
    3. TXT groundtruth'ları oku
    4. ground_truth.json oluştur
    
    Args:
        source_dir: Kaynak dizin (data/raw/TR)
        test_set_dir: Hedef dizin (tests/data/test_set)
        sample_count: Örnek sayısı
        seed: Random seed
    
    Returns:
        Test seti metadata
    """
    source_path = Path(source_dir)
    test_set_path = Path(test_set_dir)
    audio_dir = test_set_path / "audio"
    
    # Dizinleri oluştur
    audio_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*70)
    logger.info("TEST VERİ SETİ HAZIRLANIYOR")
    logger.info("="*70)
    logger.info(f"Kaynak: {source_path}")
    logger.info(f"Hedef: {test_set_path}")
    logger.info(f"Örnek sayısı: {sample_count}")
    
    # 1. Ses dosyalarını bul
    logger.info("\n[1/4] Ses dosyaları bulunuyor...")
    audio_files = find_audio_files(source_path)
    
    if len(audio_files) == 0:
        raise ValueError(f"No FLAC files found in {source_path}")
    
    # 2. Rastgele örnekler seç
    logger.info("\n[2/4] Rastgele örnekler seçiliyor...")
    selected_files = select_random_samples(audio_files, sample_count, seed)
    
    # 3. Dosyaları kopyala ve metadata topla
    logger.info("\n[3/4] Dosyalar kopyalanıyor ve metadata toplanıyor...")
    
    test_samples = []
    
    for i, audio_file in enumerate(tqdm(selected_files, desc="Processing"), start=1):
        try:
            # UUID'den basename al
            file_id = audio_file.stem  # Uzantısız dosya adı
            
            # Groundtruth TXT dosyası
            txt_file = audio_file.with_suffix('.txt')
            
            if not txt_file.exists():
                logger.warning(f"Groundtruth not found for {file_id}, skipping")
                continue
            
            # Groundtruth'u oku
            with open(txt_file, 'r', encoding='utf-8') as f:
                transcript = f.read().strip()
            
            if not transcript:
                logger.warning(f"Empty transcript for {file_id}, skipping")
                continue
            
            # Yeni dosya adı (sıralı)
            new_filename = f"sample_{i:03d}.flac"
            dest_audio = audio_dir / new_filename
            
            # Ses dosyasını kopyala
            shutil.copy2(audio_file, dest_audio)
            
            # Ses süresini hesapla
            try:
                duration = librosa.get_duration(path=str(dest_audio))
            except Exception as e:
                logger.warning(f"Could not get duration for {file_id}: {e}")
                duration = 0.0
            
            # Metadata
            sample_metadata = {
                "id": f"sample_{i:03d}",
                "audio_file": f"audio/{new_filename}",
                "original_id": file_id,
                "language": "tr",
                "duration": round(duration, 2),
                "transcript": transcript,
                "metadata": {
                    "source": "common_voice_tr",
                    "format": "flac",
                }
            }
            
            test_samples.append(sample_metadata)
        
        except Exception as e:
            logger.error(f"Failed to process {audio_file.name}: {e}")
            continue
    
    # 4. ground_truth.json oluştur
    logger.info("\n[4/4] ground_truth.json oluşturuluyor...")
    
    ground_truth = {
        "dataset_info": {
            "name": "ASR Test Set",
            "version": "1.0",
            "language": "tr",
            "source": str(source_path),
            "created_at": "2025-12-15",
            "sample_count": len(test_samples),
            "total_duration": sum(s['duration'] for s in test_samples),
            "random_seed": seed,
        },
        "test_samples": test_samples,
    }
    
    gt_file = test_set_path / "ground_truth.json"
    with open(gt_file, 'w', encoding='utf-8') as f:
        json.dump(ground_truth, f, indent=4, ensure_ascii=False)
    
    logger.success("="*70)
    logger.success(f"TEST VERİ SETİ HAZIR!")
    logger.success("="*70)
    logger.success(f"✓ Toplam örnek: {len(test_samples)}")
    logger.success(f"✓ Toplam süre: {ground_truth['dataset_info']['total_duration']:.1f} saniye")
    logger.success(f"✓ Ground truth: {gt_file}")
    logger.success(f"✓ Audio dizini: {audio_dir}")
    
    return ground_truth


def main():
    """Ana fonksiyon."""
    # Config'den ayarları al
    eval_config = config.get('evaluation', {})
    prep_config = eval_config.get('dataset_preparation', {})
    
    source_dir = prep_config.get('source_dir', './data/raw/TR')
    test_set_dir = eval_config.get('test_set_path', './tests/data/test_set')
    sample_count = prep_config.get('sample_count', 300)
    seed = prep_config.get('seed', 42)
    
    # Test setini hazırla
    try:
        ground_truth = prepare_test_set(
            source_dir=source_dir,
            test_set_dir=test_set_dir,
            sample_count=sample_count,
            seed=seed,
        )
        
        print("\n" + "="*70)
        print("İSTATİSTİKLER")
        print("="*70)
        print(f"Örnek sayısı: {ground_truth['dataset_info']['sample_count']}")
        print(f"Toplam süre: {ground_truth['dataset_info']['total_duration']:.1f}s")
        print(f"Ortalama süre: {ground_truth['dataset_info']['total_duration'] / ground_truth['dataset_info']['sample_count']:.2f}s")
        print("="*70)
        
    except Exception as e:
        logger.error(f"Test seti hazırlama başarısız: {e}")
        raise


if __name__ == "__main__":
    main()
