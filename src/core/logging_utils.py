"""
Logging Utilities Module

Provides standardized logging configuration for the trading system.
Includes:
- Structured logging with timestamps
- File and console handlers
- Log rotation
- Context managers for operation tracking
"""

from __future__ import annotations
import os
import sys
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Any
from contextlib import contextmanager
from functools import wraps
import traceback


# Custom log levels
TRACE = 5
logging.addLevelName(TRACE, "TRACE")


class ColoredFormatter(logging.Formatter):
    """Formatter with color support for console output."""
    
    COLORS = {
        "DEBUG": "\033[36m",    # Cyan
        "INFO": "\033[32m",     # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",    # Red
        "CRITICAL": "\033[35m", # Magenta
        "TRACE": "\033[90m",    # Gray
    }
    RESET = "\033[0m"
    
    def __init__(self, fmt: str = None, use_colors: bool = True):
        super().__init__(fmt or "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
        self.use_colors = use_colors and sys.stdout.isatty()
    
    def format(self, record: logging.LogRecord) -> str:
        if self.use_colors:
            color = self.COLORS.get(record.levelname, "")
            record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


class StructuredFormatter(logging.Formatter):
    """JSON-like structured formatter for machine-readable logs."""
    
    def format(self, record: logging.LogRecord) -> str:
        import json
        
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # Add extra fields
        if hasattr(record, "ticker"):
            log_data["ticker"] = record.ticker
        if hasattr(record, "stage"):
            log_data["stage"] = record.stage
        if hasattr(record, "duration_ms"):
            log_data["duration_ms"] = record.duration_ms
        
        # Add exception info
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_data)


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    log_dir: str = "outputs/logs",
    structured: bool = False,
    include_timestamp_in_filename: bool = True,
) -> logging.Logger:
    """
    Setup application-wide logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Specific log file path (overrides log_dir)
        log_dir: Directory for log files
        structured: Use JSON structured logging
        include_timestamp_in_filename: Add timestamp to auto-generated filenames
    
    Returns:
        Root logger instance
    """
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    if structured:
        console_handler.setFormatter(StructuredFormatter())
    else:
        console_handler.setFormatter(ColoredFormatter())
    
    root_logger.addHandler(console_handler)
    
    # File handler (if requested)
    if log_file or log_dir:
        if log_file:
            file_path = Path(log_file)
        else:
            log_dir_path = Path(log_dir)
            log_dir_path.mkdir(parents=True, exist_ok=True)
            
            if include_timestamp_in_filename:
                timestamp = datetime.now().strftime("%Y-%m-%d")
                filename = f"all_run_{timestamp}.log"
            else:
                filename = "latest.log"
            
            file_path = log_dir_path / filename
        
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(file_path, encoding="utf-8")
        file_handler.setLevel(level)
        
        # Always use structured format for files
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        ))
        
        root_logger.addHandler(file_handler)
    
    # Reduce noise from third-party libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("yfinance").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the given name.
    
    Args:
        name: Logger name (typically __name__)
    
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


@contextmanager
def log_operation(
    logger: logging.Logger,
    operation: str,
    ticker: Optional[str] = None,
    level: int = logging.INFO,
):
    """
    Context manager for logging operation start/end with duration.
    
    Args:
        logger: Logger instance
        operation: Operation name
        ticker: Optional ticker symbol
        level: Logging level
    
    Yields:
        Dict for storing operation metadata
    
    Example:
        with log_operation(logger, "fetch_prices", ticker="AAPL") as ctx:
            # do work
            ctx["rows_fetched"] = 100
    """
    context = {"ticker": ticker, "operation": operation}
    start_time = time.time()
    
    extra = {"ticker": ticker} if ticker else {}
    logger.log(level, f"Starting: {operation}" + (f" [{ticker}]" if ticker else ""), extra=extra)
    
    try:
        yield context
        
        duration_ms = (time.time() - start_time) * 1000
        extra["duration_ms"] = duration_ms
        
        msg = f"Completed: {operation}" + (f" [{ticker}]" if ticker else "") + f" ({duration_ms:.1f}ms)"
        logger.log(level, msg, extra=extra)
        
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        extra["duration_ms"] = duration_ms
        
        logger.error(
            f"Failed: {operation}" + (f" [{ticker}]" if ticker else "") + f" ({duration_ms:.1f}ms) - {e}",
            exc_info=True,
            extra=extra
        )
        raise


def log_exceptions(logger: logging.Logger = None, reraise: bool = True):
    """
    Decorator to log exceptions with full traceback.
    
    Args:
        logger: Logger instance (uses module logger if None)
        reraise: Whether to reraise the exception
    
    Example:
        @log_exceptions(logger)
        def risky_function():
            ...
    """
    def decorator(func):
        nonlocal logger
        if logger is None:
            logger = logging.getLogger(func.__module__)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(
                    f"Exception in {func.__name__}: {e}",
                    exc_info=True
                )
                if reraise:
                    raise
                return None
        
        return wrapper
    return decorator


class ProgressLogger:
    """
    Helper for logging progress of long-running operations.
    
    Example:
        progress = ProgressLogger(logger, total=1000, operation="screening")
        for i, ticker in enumerate(tickers):
            # process ticker
            progress.update(i + 1)
        progress.finish()
    """
    
    def __init__(
        self,
        logger: logging.Logger,
        total: int,
        operation: str = "processing",
        interval: int = 10,  # Log every N percent
    ):
        self.logger = logger
        self.total = total
        self.operation = operation
        self.interval = interval
        self.start_time = time.time()
        self.last_logged_pct = -interval
    
    def update(self, current: int, extra_msg: str = ""):
        """Update progress and log if threshold reached."""
        if self.total == 0:
            return
        
        pct = (current / self.total) * 100
        
        if pct >= self.last_logged_pct + self.interval:
            elapsed = time.time() - self.start_time
            rate = current / elapsed if elapsed > 0 else 0
            eta = (self.total - current) / rate if rate > 0 else 0
            
            msg = f"{self.operation}: {current}/{self.total} ({pct:.0f}%)"
            if extra_msg:
                msg += f" | {extra_msg}"
            msg += f" | ETA: {eta:.0f}s"
            
            self.logger.info(msg)
            self.last_logged_pct = int(pct / self.interval) * self.interval
    
    def finish(self, summary: str = ""):
        """Log completion."""
        elapsed = time.time() - self.start_time
        rate = self.total / elapsed if elapsed > 0 else 0
        
        msg = f"{self.operation}: Complete ({self.total} items in {elapsed:.1f}s, {rate:.1f}/s)"
        if summary:
            msg += f" | {summary}"
        
        self.logger.info(msg)


# Convenience function for backward compatibility
def setup_basic_logging(debug: bool = False, log_file: Optional[str] = None):
    """Simple logging setup for scripts."""
    level = logging.DEBUG if debug else logging.INFO
    setup_logging(
        level=level,
        log_file=Path(log_file) if log_file else None,
    )
