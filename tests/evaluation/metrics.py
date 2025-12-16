"""
Metrik hesaplama modülü.
WER, CER, RTF ve detaylı hata analizi fonksiyonları.
"""

from typing import Dict, Optional
from jiwer import wer, cer, process_words
from loguru import logger


def calculate_wer(reference: str, hypothesis: str) -> float:
    """
    Word Error Rate (WER) hesapla.
    
    WER = (Substitutions + Deletions + Insertions) / Total Words
    
    Args:
        reference: Ground truth metin
        hypothesis: Model tarafından üretilen metin
    
    Returns:
        WER değeri (0.0 - 1.0 arası, düşük = iyi)
    
    Example:
        >>> calculate_wer("hello world", "hello word")
        0.5  # 1 hata / 2 kelime
    """
    try:
        if not reference or not hypothesis:
            logger.warning("Empty reference or hypothesis, returning WER=1.0")
            return 1.0
        
        wer_score = wer(reference, hypothesis)
        return wer_score
    
    except Exception as e:
        logger.error(f"WER calculation failed: {e}")
        return 1.0


def calculate_cer(reference: str, hypothesis: str) -> float:
    """
    Character Error Rate (CER) hesapla.
    
    CER = (Substitutions + Deletions + Insertions) / Total Characters
    
    Args:
        reference: Ground truth metin
        hypothesis: Model tarafından üretilen metin
    
    Returns:
        CER değeri (0.0 - 1.0 arası, düşük = iyi)
    
    Example:
        >>> calculate_cer("hello", "helo")
        0.2  # 1 hata / 5 karakter
    """
    try:
        if not reference or not hypothesis:
            logger.warning("Empty reference or hypothesis, returning CER=1.0")
            return 1.0
        
        cer_score = cer(reference, hypothesis)
        return cer_score
    
    except Exception as e:
        logger.error(f"CER calculation failed: {e}")
        return 1.0


def calculate_rtf(audio_duration: float, process_time: float) -> float:
    """
    Real-Time Factor (RTF) hesapla.
    
    RTF = Processing Time / Audio Duration
    
    RTF < 1.0 = Gerçek zamanlı işleme mümkün
    RTF = 0.5 = 10 saniyelik ses 5 saniyede işlenir
    RTF = 2.0 = 10 saniyelik ses 20 saniyede işlenir
    
    Args:
        audio_duration: Ses dosyasının süresi (saniye)
        process_time: İşlem süresi (saniye)
    
    Returns:
        RTF değeri (düşük = hızlı)
    
    Example:
        >>> calculate_rtf(10.0, 5.0)
        0.5  # 2x daha hızlı
    """
    if audio_duration <= 0:
        logger.warning("Invalid audio duration, returning RTF=inf")
        return float('inf')
    
    rtf = process_time / audio_duration
    return rtf


def detailed_error_analysis(reference: str, hypothesis: str) -> Dict[str, float]:
    """
    Detaylı hata analizi yap.
    
    Substitution, Deletion, Insertion sayılarını ve
    ek metrikleri (MER, WIL) hesapla.
    
    Args:
        reference: Ground truth metin
        hypothesis: Model tarafından üretilen metin
    
    Returns:
        Dict içinde:
            - wer: Word Error Rate
            - cer: Character Error Rate
            - substitutions: Yanlış kelime sayısı
            - deletions: Eksik kelime sayısı
            - insertions: Fazla kelime sayısı
            - hits: Doğru kelime sayısı
            - mer: Match Error Rate
            - wil: Word Information Lost
            - total_words: Toplam kelime sayısı
    
    Example:
        >>> detailed_error_analysis("hello world", "hello word")
        {
            'wer': 0.5,
            'substitutions': 1,
            'deletions': 0,
            'insertions': 0,
            'hits': 1,
            ...
        }
    """
    try:
        if not reference or not hypothesis:
            logger.warning("Empty reference or hypothesis")
            return {
                'wer': 1.0,
                'cer': 1.0,
                'substitutions': 0,
                'deletions': 0,
                'insertions': 0,
                'hits': 0,
                'mer': 1.0,
                'wil': 1.0,
                'total_words': len(reference.split()) if reference else 0,
            }
        
        # jiwer ile detaylı analiz
        output = process_words(reference, hypothesis)
        
        # CER'i ayrıca hesapla
        cer_score = calculate_cer(reference, hypothesis)
        
        result = {
            'wer': float(output.wer),
            'cer': cer_score,
            'substitutions': int(output.substitutions),
            'deletions': int(output.deletions),
            'insertions': int(output.insertions),
            'hits': int(output.hits),
            'mer': float(output.mer),  # Match Error Rate
            'wil': float(output.wil),  # Word Information Lost
            'total_words': len(reference.split()),
        }
        
        logger.debug(f"Error analysis: WER={result['wer']:.2%}, "
                    f"S={result['substitutions']}, D={result['deletions']}, I={result['insertions']}")
        
        return result
    
    except Exception as e:
        logger.error(f"Detailed error analysis failed: {e}")
        return {
            'wer': 1.0,
            'cer': 1.0,
            'substitutions': 0,
            'deletions': 0,
            'insertions': 0,
            'hits': 0,
            'mer': 1.0,
            'wil': 1.0,
            'total_words': len(reference.split()) if reference else 0,
        }


def calculate_accuracy(reference: str, hypothesis: str) -> float:
    """
    Accuracy hesapla (1 - WER).
    
    Args:
        reference: Ground truth metin
        hypothesis: Model tarafından üretilen metin
    
    Returns:
        Accuracy değeri (0.0 - 1.0 arası, yüksek = iyi)
    """
    wer_score = calculate_wer(reference, hypothesis)
    return max(0.0, 1.0 - wer_score)


def format_metrics_summary(metrics: Dict[str, float]) -> str:
    """
    Metrik özetini formatla (logging/display için).
    
    Args:
        metrics: Metrik dictionary
    
    Returns:
        Formatlanmış string
    
    Example:
        >>> metrics = {'wer': 0.15, 'cer': 0.08, 'rtf': 0.45}
        >>> print(format_metrics_summary(metrics))
        WER: 15.00% | CER: 8.00% | RTF: 0.45
    """
    parts = []
    
    if 'wer' in metrics:
        parts.append(f"WER: {metrics['wer']:.2%}")
    
    if 'cer' in metrics:
        parts.append(f"CER: {metrics['cer']:.2%}")
    
    if 'rtf' in metrics:
        parts.append(f"RTF: {metrics['rtf']:.2f}")
    
    if 'substitutions' in metrics:
        parts.append(f"S: {metrics['substitutions']}")
    
    if 'deletions' in metrics:
        parts.append(f"D: {metrics['deletions']}")
    
    if 'insertions' in metrics:
        parts.append(f"I: {metrics['insertions']}")
    
    return " | ".join(parts)


# Türkçe için normalize edilmiş metin karşılaştırması
def normalize_turkish_text(text: str) -> str:
    """
    Türkçe metin normalizasyonu (WER hesabından önce).
    
    - Küçük harfe çevir
    - Fazla boşlukları temizle
    - Noktalama işaretlerini kaldır (opsiyonel)
    
    Args:
        text: Ham metin
    
    Returns:
        Normalize edilmiş metin
    """
    import re
    
    # Küçük harfe çevir (Türkçe karakterler korunur)
    text = text.lower()
    
    # Fazla boşlukları temizle
    text = re.sub(r'\s+', ' ', text)
    
    # Baş/son boşlukları kaldır
    text = text.strip()
    
    return text


def calculate_wer_normalized(reference: str, hypothesis: str) -> float:
    """
    Normalize edilmiş metinler üzerinden WER hesapla.
    
    Args:
        reference: Ground truth metin
        hypothesis: Model tarafından üretilen metin
    
    Returns:
        WER değeri
    """
    ref_normalized = normalize_turkish_text(reference)
    hyp_normalized = normalize_turkish_text(hypothesis)
    
    return calculate_wer(ref_normalized, hyp_normalized)
