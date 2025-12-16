"""
Sonuç raporlama modülü.
Benchmark sonuçlarını JSON, CSV, Markdown formatlarında kaydeder.
"""

import json
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd
from loguru import logger


class ReportGenerator:
    """
    Benchmark sonuç raporlayıcı.
    
    Desteklenen formatlar:
    - JSON (detaylı sonuçlar)
    - CSV (tablo formatı)
    - Markdown (insan okunabilir)
    - LaTeX (akademik rapor için - opsiyonel)
    """
    
    def __init__(self, results_dir: str = "./tests/data/results"):
        """
        Args:
            results_dir: Sonuçların kaydedileceği dizin
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        logger.debug(f"ReportGenerator initialized - results_dir: {self.results_dir}")
    
    def save_results(
        self,
        results: Dict[str, Any],
        formats: List[str] = ['json', 'csv'],
        base_filename: Optional[str] = None,
    ) -> Dict[str, Path]:
        """
        Sonuçları belirtilen formatlarda kaydet.
        
        Args:
            results: Benchmark sonuçları
            formats: Kaydedilecek formatlar ['json', 'csv', 'markdown', 'latex']
            base_filename: Dosya adı (None = otomatik timestamp)
        
        Returns:
            Format -> dosya yolu mapping
        """
        if base_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = f"benchmark_{timestamp}"
        
        saved_files = {}
        
        for fmt in formats:
            if fmt == 'json':
                filepath = self._save_json(results, base_filename)
                saved_files['json'] = filepath
            
            elif fmt == 'csv':
                filepath = self._save_csv(results, base_filename)
                saved_files['csv'] = filepath
            
            elif fmt == 'markdown':
                filepath = self._save_markdown(results, base_filename)
                saved_files['markdown'] = filepath
            
            elif fmt == 'latex':
                filepath = self._save_latex(results, base_filename)
                saved_files['latex'] = filepath
            
            else:
                logger.warning(f"Unknown format: {fmt}")
        
        logger.info(f"Results saved: {', '.join(str(p) for p in saved_files.values())}")
        return saved_files
    
    def _save_json(self, results: Dict[str, Any], base_filename: str) -> Path:
        """JSON formatında kaydet."""
        filepath = self.results_dir / f"{base_filename}.json"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.debug(f"JSON saved: {filepath}")
        return filepath
    
    def _save_csv(self, results: Dict[str, Any], base_filename: str) -> Path:
        """CSV formatında kaydet (pandas kullanarak)."""
        filepath = self.results_dir / f"{base_filename}.csv"
        
        # results yapısından DataFrame oluştur
        rows = []
        
        # Her test sonucunu bir satıra çevir
        for result in results.get('results', []):
            row = {}
            
            # Model bilgileri
            model_info = result.get('model', {})
            row['model_name'] = model_info.get('name', 'unknown')
            row['model_variant'] = model_info.get('variant', 'unknown')
            row['compute_type'] = model_info.get('compute_type', 'unknown')
            
            # Preprocessing
            preprocessing = result.get('preprocessing', {})
            row['preprocessing_enabled'] = preprocessing.get('enabled', False)
            
            # Metrikler (ortalama değerler)
            metrics = result.get('metrics', {})
            
            # Dil bazlı metrikler
            for lang in ['turkish', 'english']:
                if lang in metrics:
                    lang_metrics = metrics[lang]
                    row[f'{lang}_wer'] = lang_metrics.get('wer', None)
                    row[f'{lang}_cer'] = lang_metrics.get('cer', None)
                    row[f'{lang}_rtf'] = lang_metrics.get('avg_rtf', None)
            
            # Kaynak kullanımı
            resources = result.get('resources', {})
            row['cpu_percent'] = resources.get('avg_cpu_percent', None)
            row['memory_mb'] = resources.get('peak_memory_mb', None)
            row['latency_ms'] = resources.get('avg_latency_ms', None)
            
            # Test bilgileri
            row['test_set_size'] = result.get('test_set_size', None)
            row['timestamp'] = result.get('timestamp', None)
            
            rows.append(row)
        
        # DataFrame oluştur ve kaydet
        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False, encoding='utf-8')
        
        logger.debug(f"CSV saved: {filepath} ({len(rows)} rows)")
        return filepath
    
    def _save_markdown(self, results: Dict[str, Any], base_filename: str) -> Path:
        """Markdown formatında kaydet (insan okunabilir)."""
        filepath = self.results_dir / f"{base_filename}.md"
        
        lines = []
        lines.append("# ASR Benchmark Sonuçları\n")
        
        # Genel bilgiler
        benchmark_info = results.get('benchmark_info', {})
        lines.append("## Benchmark Bilgileri\n")
        lines.append(f"- **Tarih:** {benchmark_info.get('timestamp', 'N/A')}")
        lines.append(f"- **Test Set Boyutu:** {benchmark_info.get('test_set_size', 'N/A')}")
        lines.append(f"- **Toplam Süre:** {benchmark_info.get('total_duration', 'N/A')}")
        lines.append("")
        
        # Sistem bilgileri (varsa)
        if 'system_info' in results:
            system_info = results['system_info']
            lines.append("## Sistem Bilgileri\n")
            
            platform_info = system_info.get('platform', {})
            lines.append(f"- **OS:** {platform_info.get('system', 'N/A')} {platform_info.get('release', '')}")
            lines.append(f"- **İşlemci:** {platform_info.get('processor', 'N/A')}")
            
            cpu_info = system_info.get('cpu', {})
            lines.append(f"- **CPU Çekirdek:** {cpu_info.get('logical_cores', 'N/A')}")
            
            memory_info = system_info.get('memory', {})
            lines.append(f"- **RAM:** {memory_info.get('total_gb', 'N/A'):.1f}GB")
            lines.append("")
        
        # Sonuç tablosu
        lines.append("## Sonuçlar (Normalized)\n")
        lines.append("| Model | Variant | Preprocessing | WER (TR) | WER Raw | CER (TR) | RTF | CPU % | Memory (MB) |")
        lines.append("|-------|---------|---------------|----------|---------|----------|-----|-------|-------------|")
        
        for result in results.get('results', []):
            model_info = result.get('model', {})
            model_name = model_info.get('name', 'unknown')
            variant = model_info.get('variant', 'unknown')
            
            preprocessing = result.get('preprocessing', {})
            prep_enabled = "✓" if preprocessing.get('enabled', False) else "✗"
            
            metrics = result.get('metrics', {})
            tr_metrics = metrics.get('turkish', {})
            wer = tr_metrics.get('wer', 0.0)
            cer = tr_metrics.get('cer', 0.0)
            rtf = tr_metrics.get('avg_rtf', 0.0)
            
            # Raw WER (varsa)
            wer_raw = tr_metrics.get('wer_raw', wer)
            wer_raw_str = f"{wer_raw:.2%}" if wer_raw != wer else "-"
            
            resources = result.get('resources', {})
            cpu = resources.get('avg_cpu_percent', 0.0)
            memory = resources.get('peak_memory_mb', 0.0)
            
            lines.append(f"| {model_name} | {variant} | {prep_enabled} | "
                        f"{wer:.2%} | {wer_raw_str} | {cer:.2%} | {rtf:.2f} | {cpu:.1f} | {memory:.1f} |")
        
        lines.append("")
        lines.append("**Not:** WER (normalized) = küçük harf, noktalama temizleme ile hesaplanmış")
        lines.append("")
        
        # En iyi sonuçlar
        lines.append("## En İyi Sonuçlar\n")
        
        if results.get('results'):
            # WER'e göre sırala
            sorted_results = sorted(
                results['results'],
                key=lambda x: x.get('metrics', {}).get('turkish', {}).get('wer', 1.0)
            )
            
            best = sorted_results[0]
            model_info = best.get('model', {})
            tr_metrics = best.get('metrics', {}).get('turkish', {})
            
            lines.append(f"- **En Düşük WER:** {model_info.get('name')} {model_info.get('variant')} "
                        f"({tr_metrics.get('wer', 0.0):.2%})")
            
            # RTF'ye göre en hızlı
            sorted_by_rtf = sorted(
                results['results'],
                key=lambda x: x.get('metrics', {}).get('turkish', {}).get('avg_rtf', 999.0)
            )
            
            fastest = sorted_by_rtf[0]
            model_info = fastest.get('model', {})
            rtf = fastest.get('metrics', {}).get('turkish', {}).get('avg_rtf', 0.0)
            
            lines.append(f"- **En Hızlı Model:** {model_info.get('name')} {model_info.get('variant')} "
                        f"(RTF: {rtf:.2f})")
        
        lines.append("")
        lines.append("---")
        lines.append(f"*Rapor oluşturma zamanı: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
        
        # Dosyaya yaz
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        logger.debug(f"Markdown saved: {filepath}")
        return filepath
    
    def _save_latex(self, results: Dict[str, Any], base_filename: str) -> Path:
        """LaTeX tablo formatında kaydet (akademik rapor için)."""
        filepath = self.results_dir / f"{base_filename}.tex"
        
        lines = []
        lines.append("% ASR Benchmark Sonuçları (LaTeX Tablo)")
        lines.append("% Faz 8 akademik raporu için")
        lines.append("")
        lines.append("\\begin{table}[htbp]")
        lines.append("\\centering")
        lines.append("\\caption{ASR Model Karşılaştırma Sonuçları}")
        lines.append("\\label{tab:asr_benchmark}")
        lines.append("\\begin{tabular}{llccccc}")
        lines.append("\\hline")
        lines.append("Model & Variant & Prep. & WER (\\%) & CER (\\%) & RTF & CPU (\\%) \\\\")
        lines.append("\\hline")
        
        for result in results.get('results', []):
            model_info = result.get('model', {})
            model_name = model_info.get('name', 'unknown')
            variant = model_info.get('variant', 'unknown')
            
            preprocessing = result.get('preprocessing', {})
            prep_enabled = "\\checkmark" if preprocessing.get('enabled', False) else ""
            
            metrics = result.get('metrics', {})
            tr_metrics = metrics.get('turkish', {})
            wer = tr_metrics.get('wer', 0.0) * 100
            cer = tr_metrics.get('cer', 0.0) * 100
            rtf = tr_metrics.get('avg_rtf', 0.0)
            
            resources = result.get('resources', {})
            cpu = resources.get('avg_cpu_percent', 0.0)
            
            lines.append(f"{model_name} & {variant} & {prep_enabled} & "
                        f"{wer:.2f} & {cer:.2f} & {rtf:.2f} & {cpu:.1f} \\\\")
        
        lines.append("\\hline")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}")
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        logger.debug(f"LaTeX saved: {filepath}")
        return filepath
    
    def print_summary(self, results: Dict[str, Any]) -> None:
        """Konsola özet yazdır."""
        print("\n" + "="*70)
        print("ASR BENCHMARK SONUÇLARI")
        print("="*70)
        
        # Benchmark bilgileri
        benchmark_info = results.get('benchmark_info', {})
        print(f"\nTest Set: {benchmark_info.get('test_set_size', 'N/A')} örnek")
        print(f"Tarih: {benchmark_info.get('timestamp', 'N/A')}")
        print(f"Süre: {benchmark_info.get('total_duration', 'N/A')}")
        
        # Sonuçlar
        print("\n" + "-"*70)
        print(f"{'Model':<20} {'Variant':<10} {'WER':<8} {'RTF':<8} {'Memory':<12}")
        print("-"*70)
        
        for result in results.get('results', []):
            model_info = result.get('model', {})
            model_name = model_info.get('name', 'unknown')
            variant = model_info.get('variant', 'unknown')
            
            metrics = result.get('metrics', {})
            tr_metrics = metrics.get('turkish', {})
            wer = tr_metrics.get('wer', 0.0)
            rtf = tr_metrics.get('avg_rtf', 0.0)
            
            resources = result.get('resources', {})
            memory = resources.get('peak_memory_mb', 0.0)
            
            print(f"{model_name:<20} {variant:<10} {wer:>6.2%} {rtf:>7.2f} {memory:>10.1f}MB")
        
        print("="*70 + "\n")
    
    def export_for_plotting(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Grafik çizimi için data hazırla (matplotlib/seaborn).
        
        Returns:
            Dict içinde plotting için hazır datalar
        """
        plot_data = {
            'wer_comparison': [],
            'speed_comparison': [],
            'resource_usage': [],
        }
        
        for result in results.get('results', []):
            model_info = result.get('model', {})
            model_label = f"{model_info.get('name', 'unknown')}-{model_info.get('variant', 'unknown')}"
            
            metrics = result.get('metrics', {})
            tr_metrics = metrics.get('turkish', {})
            
            # WER comparison
            plot_data['wer_comparison'].append({
                'model': model_label,
                'wer': tr_metrics.get('wer', 0.0),
                'cer': tr_metrics.get('cer', 0.0),
            })
            
            # Speed comparison
            plot_data['speed_comparison'].append({
                'model': model_label,
                'rtf': tr_metrics.get('avg_rtf', 0.0),
            })
            
            # Resource usage
            resources = result.get('resources', {})
            plot_data['resource_usage'].append({
                'model': model_label,
                'cpu_percent': resources.get('avg_cpu_percent', 0.0),
                'memory_mb': resources.get('peak_memory_mb', 0.0),
            })
        
        return plot_data


def generate_comparison_table(
    results: Dict[str, Any],
    format: str = 'markdown'
) -> str:
    """
    Faz 8 için karşılaştırma tablosu oluştur.
    
    Args:
        results: Benchmark sonuçları
        format: 'markdown', 'latex', 'html'
    
    Returns:
        Formatlanmış tablo string
    """
    generator = ReportGenerator()
    
    if format == 'markdown':
        temp_file = generator._save_markdown(results, 'temp')
        with open(temp_file, 'r', encoding='utf-8') as f:
            content = f.read()
        temp_file.unlink()  # Geçici dosyayı sil
        return content
    
    elif format == 'latex':
        temp_file = generator._save_latex(results, 'temp')
        with open(temp_file, 'r', encoding='utf-8') as f:
            content = f.read()
        temp_file.unlink()
        return content
    
    else:
        raise ValueError(f"Unsupported format: {format}")
