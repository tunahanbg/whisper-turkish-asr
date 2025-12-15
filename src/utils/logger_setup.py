"""
Logging configuration using loguru.
DRY: Merkezi logging sistemi.
"""

import sys
from pathlib import Path
from typing import Optional
from loguru import logger

from config import config


def setup_logger(
    log_file: Optional[str] = None,
    level: Optional[str] = None,
    rotation: Optional[str] = None,
    retention: Optional[str] = None,
) -> None:
    """
    Loguru logger'ı yapılandır.
    Config dosyasından ayarları alır, parametrelerle override edilebilir.
    
    Args:
        log_file: Log dosyası yolu (opsiyonel)
        level: Log seviyesi (DEBUG, INFO, WARNING, ERROR)
        rotation: Log rotation ayarı (örn: "100 MB", "1 day")
        retention: Log saklama süresi (örn: "1 week", "10 days")
    """
    # Config'den ayarları al
    log_config = config.get('logging', {})
    
    level = level or log_config.get('level', 'INFO')
    log_file = log_file or log_config.get('file', './logs/asr_system.log')
    rotation = rotation or log_config.get('rotation', '100 MB')
    retention = retention or log_config.get('retention', '1 week')
    log_format = log_config.get('format', 
        "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}")
    console = log_config.get('console', True)
    
    # Mevcut handler'ları temizle
    logger.remove()
    
    # Console handler (opsiyonel)
    if console:
        logger.add(
            sys.stderr,
            format=log_format,
            level=level,
            colorize=True,
        )
    
    # File handler
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.add(
        log_file,
        format=log_format,
        level=level,
        rotation=rotation,
        retention=retention,
        compression="zip",
        encoding="utf-8",
    )
    
    logger.info(f"Logger initialized - Level: {level}, File: {log_file}")


# Auto-initialize logger
setup_logger()

