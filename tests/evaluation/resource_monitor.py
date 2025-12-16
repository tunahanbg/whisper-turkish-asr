"""
Kaynak kullanımı izleme modülü.
CPU, memory, disk kullanımını izler ve raporlar.
"""

import time
import psutil
from typing import Dict, Optional, Any
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from loguru import logger


@dataclass
class ResourceStats:
    """Kaynak kullanım istatistikleri."""
    
    # Süre
    duration_seconds: float
    
    # CPU
    cpu_percent_avg: float
    cpu_percent_max: float
    cpu_count: int
    
    # Memory
    memory_mb_start: float
    memory_mb_end: float
    memory_mb_peak: float
    memory_mb_increase: float
    memory_percent_avg: float
    
    # Disk (opsiyonel)
    disk_io_read_mb: Optional[float] = None
    disk_io_write_mb: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Dict'e çevir."""
        return asdict(self)
    
    def __str__(self) -> str:
        """Human-readable format."""
        return (f"Duration: {self.duration_seconds:.2f}s | "
                f"CPU: {self.cpu_percent_avg:.1f}% (max: {self.cpu_percent_max:.1f}%) | "
                f"Memory: {self.memory_mb_peak:.1f}MB peak "
                f"(+{self.memory_mb_increase:.1f}MB)")


class ResourceMonitor:
    """
    Kaynak kullanımı izleyici.
    
    Context manager olarak kullanılabilir:
        with ResourceMonitor() as monitor:
            # kod
        
        stats = monitor.get_stats()
    """
    
    def __init__(self, sample_interval: float = 0.1):
        """
        Args:
            sample_interval: CPU örnekleme aralığı (saniye)
        """
        self.sample_interval = sample_interval
        self.process = psutil.Process()
        
        # Başlangıç değerleri
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        
        self.start_memory_mb: Optional[float] = None
        self.end_memory_mb: Optional[float] = None
        self.peak_memory_mb: Optional[float] = None
        
        self.cpu_samples = []
        self.memory_samples = []
        
        # Disk I/O (opsiyonel)
        self.start_disk_io: Optional[Any] = None
        self.end_disk_io: Optional[Any] = None
        
        logger.debug(f"ResourceMonitor initialized (interval={sample_interval}s)")
    
    def start(self) -> None:
        """İzlemeyi başlat."""
        self.start_time = time.time()
        
        # Memory
        mem_info = self.process.memory_info()
        self.start_memory_mb = mem_info.rss / 1024 / 1024
        self.peak_memory_mb = self.start_memory_mb
        
        # CPU (ilk okuma genelde 0 döner, bir kere okuyalım)
        self.process.cpu_percent()
        
        # Disk I/O
        try:
            self.start_disk_io = self.process.io_counters()
        except (psutil.AccessDenied, AttributeError):
            self.start_disk_io = None
        
        logger.debug(f"Monitoring started - Memory: {self.start_memory_mb:.1f}MB")
    
    def sample(self) -> None:
        """Anlık ölçüm al (döngü içinde çağrılabilir)."""
        # CPU
        cpu = self.process.cpu_percent()
        self.cpu_samples.append(cpu)
        
        # Memory
        mem_info = self.process.memory_info()
        mem_mb = mem_info.rss / 1024 / 1024
        self.memory_samples.append(mem_mb)
        
        # Peak memory güncelle
        if mem_mb > self.peak_memory_mb:
            self.peak_memory_mb = mem_mb
    
    def stop(self) -> None:
        """İzlemeyi durdur."""
        self.end_time = time.time()
        
        # Son örnekleri al
        self.sample()
        
        # Memory
        mem_info = self.process.memory_info()
        self.end_memory_mb = mem_info.rss / 1024 / 1024
        
        # Disk I/O
        try:
            self.end_disk_io = self.process.io_counters()
        except (psutil.AccessDenied, AttributeError):
            self.end_disk_io = None
        
        logger.debug(f"Monitoring stopped - Duration: {self.end_time - self.start_time:.2f}s")
    
    def get_stats(self) -> ResourceStats:
        """İstatistikleri hesapla ve döndür."""
        if self.start_time is None or self.end_time is None:
            raise RuntimeError("Monitor not started/stopped properly")
        
        # Süre
        duration = self.end_time - self.start_time
        
        # CPU
        cpu_avg = sum(self.cpu_samples) / len(self.cpu_samples) if self.cpu_samples else 0.0
        cpu_max = max(self.cpu_samples) if self.cpu_samples else 0.0
        cpu_count = psutil.cpu_count()
        
        # Memory
        memory_avg_percent = (self.process.memory_percent() 
                              if hasattr(self.process, 'memory_percent') else 0.0)
        memory_increase = self.end_memory_mb - self.start_memory_mb
        
        # Disk I/O
        disk_read_mb = None
        disk_write_mb = None
        if self.start_disk_io and self.end_disk_io:
            disk_read_mb = (self.end_disk_io.read_bytes - 
                           self.start_disk_io.read_bytes) / 1024 / 1024
            disk_write_mb = (self.end_disk_io.write_bytes - 
                            self.start_disk_io.write_bytes) / 1024 / 1024
        
        stats = ResourceStats(
            duration_seconds=duration,
            cpu_percent_avg=cpu_avg,
            cpu_percent_max=cpu_max,
            cpu_count=cpu_count,
            memory_mb_start=self.start_memory_mb,
            memory_mb_end=self.end_memory_mb,
            memory_mb_peak=self.peak_memory_mb,
            memory_mb_increase=memory_increase,
            memory_percent_avg=memory_avg_percent,
            disk_io_read_mb=disk_read_mb,
            disk_io_write_mb=disk_write_mb,
        )
        
        logger.info(f"Resource stats: {stats}")
        return stats
    
    def __enter__(self) -> 'ResourceMonitor':
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.stop()


@contextmanager
def monitor_resources(sample_interval: float = 0.1):
    """
    Basit context manager wrapper.
    
    Usage:
        with monitor_resources() as monitor:
            # kod
        
        stats = monitor.get_stats()
    
    Args:
        sample_interval: CPU örnekleme aralığı
    
    Yields:
        ResourceMonitor instance
    """
    monitor = ResourceMonitor(sample_interval=sample_interval)
    monitor.start()
    
    try:
        yield monitor
    finally:
        monitor.stop()


def get_system_info() -> Dict[str, Any]:
    """
    Sistem bilgilerini al (benchmark başlangıcında kaydetmek için).
    
    Returns:
        Sistem bilgileri dict
    """
    import platform
    
    # CPU
    cpu_freq = psutil.cpu_freq()
    
    # Memory
    mem = psutil.virtual_memory()
    
    # Disk
    disk = psutil.disk_usage('/')
    
    info = {
        'platform': {
            'system': platform.system(),
            'release': platform.release(),
            'version': platform.version(),
            'machine': platform.machine(),
            'processor': platform.processor(),
        },
        'cpu': {
            'physical_cores': psutil.cpu_count(logical=False),
            'logical_cores': psutil.cpu_count(logical=True),
            'max_frequency_mhz': cpu_freq.max if cpu_freq else None,
            'current_frequency_mhz': cpu_freq.current if cpu_freq else None,
        },
        'memory': {
            'total_gb': mem.total / 1024 / 1024 / 1024,
            'available_gb': mem.available / 1024 / 1024 / 1024,
            'used_percent': mem.percent,
        },
        'disk': {
            'total_gb': disk.total / 1024 / 1024 / 1024,
            'free_gb': disk.free / 1024 / 1024 / 1024,
            'used_percent': disk.percent,
        },
    }
    
    logger.debug(f"System info: {platform.system()} {platform.machine()}, "
                f"{info['cpu']['logical_cores']} cores, "
                f"{info['memory']['total_gb']:.1f}GB RAM")
    
    return info


# Kullanım örneği
if __name__ == "__main__":
    import sys
    
    # Sistem bilgisi
    print("System Info:")
    print("-" * 50)
    info = get_system_info()
    print(f"OS: {info['platform']['system']} {info['platform']['release']}")
    print(f"CPU: {info['cpu']['logical_cores']} cores @ {info['cpu']['current_frequency_mhz']:.0f}MHz")
    print(f"Memory: {info['memory']['total_gb']:.1f}GB (available: {info['memory']['available_gb']:.1f}GB)")
    print()
    
    # Test monitoring
    print("Testing resource monitoring...")
    print("-" * 50)
    
    with monitor_resources() as monitor:
        # Dummy iş yükü
        result = sum(i**2 for i in range(1000000))
        time.sleep(0.5)
    
    stats = monitor.get_stats()
    print(stats)
    print()
    print(f"Peak memory: {stats.memory_mb_peak:.1f}MB")
    print(f"Memory increase: {stats.memory_mb_increase:.1f}MB")
    print(f"Average CPU: {stats.cpu_percent_avg:.1f}%")
