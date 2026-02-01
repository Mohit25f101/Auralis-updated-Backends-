# ============================================
# app/core/logger.py - Logging Configuration
# ============================================

import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

try:
    from loguru import logger as loguru_logger
    USE_LOGURU = True
except ImportError:
    USE_LOGURU = False


class AuralisLogger:
    """Centralized logging for Auralis Ultimate."""
    
    def __init__(self, name: str = "auralis", log_dir: Optional[Path] = None):
        self.name = name
        self.log_dir = log_dir or Path("logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        if USE_LOGURU:
            self._setup_loguru()
        else:
            self._setup_standard()
    
    def _setup_loguru(self):
        """Configure loguru logger."""
        loguru_logger.remove()
        
        # Console output with colors
        loguru_logger.add(
            sys.stdout,
            format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> | {message}",
            level="INFO",
            colorize=True
        )
        
        # File output
        log_file = self.log_dir / f"auralis_{datetime.now().strftime('%Y%m%d')}.log"
        loguru_logger.add(
            str(log_file),
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name} | {message}",
            level="DEBUG",
            rotation="10 MB",
            retention="7 days",
            compression="zip"
        )
        
        self._logger = loguru_logger
    
    def _setup_standard(self):
        """Configure standard logging."""
        self._logger = logging.getLogger(self.name)
        self._logger.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_format)
        
        # File handler
        log_file = self.log_dir / f"auralis_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(str(log_file))
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
        )
        file_handler.setFormatter(file_format)
        
        self._logger.addHandler(console_handler)
        self._logger.addHandler(file_handler)
    
    def debug(self, message: str, **kwargs):
        if USE_LOGURU:
            self._logger.debug(message, **kwargs)
        else:
            self._logger.debug(message)
    
    def info(self, message: str, **kwargs):
        if USE_LOGURU:
            self._logger.info(message, **kwargs)
        else:
            self._logger.info(message)
    
    def warning(self, message: str, **kwargs):
        if USE_LOGURU:
            self._logger.warning(message, **kwargs)
        else:
            self._logger.warning(message)
    
    def error(self, message: str, **kwargs):
        if USE_LOGURU:
            self._logger.error(message, **kwargs)
        else:
            self._logger.error(message)
    
    def critical(self, message: str, **kwargs):
        if USE_LOGURU:
            self._logger.critical(message, **kwargs)
        else:
            self._logger.critical(message)


# Global logger instance
_logger_instance: Optional[AuralisLogger] = None


def get_logger(name: str = "auralis") -> AuralisLogger:
    """Get or create logger instance."""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = AuralisLogger(name)
    return _logger_instance