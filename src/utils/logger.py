#!/usr/bin/env python3
"""
Production-grade logging system for TDA experiments.
"""

import logging
import json
import time
import psutil
import traceback
import uuid
import sys
import io
import os
import platform
import functools
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple, Generator
from contextlib import contextmanager

# --- Helpers for safe console output on Windows --------------------------------

def _utf8_stream(stream):
    """Wrap a stream so writes are UTF-8 with replacement (no crashes)."""
    if isinstance(stream, io.TextIOBase) and getattr(stream, "encoding", "").lower() == "utf-8":
        return stream
    if hasattr(stream, "buffer"):
        return io.TextIOWrapper(stream.buffer, encoding="utf-8", errors="replace")
    return stream

# Treat Windows consoles as emoji-unsafe unless user opted into UTF-8.
_EMOJI_OK = (
    platform.system() != "Windows"
    or os.environ.get("PYTHONUTF8") == "1"
    or os.environ.get("WT_SESSION")  # Windows Terminal usually supports UTF-8
)

# Allow forcing ASCII-only console logs regardless of OS (optional).
_FORCE_ASCII = os.environ.get("TDA_ASCII_LOGS") == "1"

class _ConsoleUnicodeFilter(logging.Filter):
    """For console handler only: strip non-ASCII when emojis would crash."""
    def filter(self, record: logging.LogRecord) -> bool:
        if _FORCE_ASCII or not _EMOJI_OK:
            try:
                # Ensure we strip only for console; other handlers (file/json) still get full Unicode.
                record.msg = str(record.getMessage()).encode("ascii", "ignore").decode("ascii")
                record.args = ()
            except Exception:
                # Best effort; never block logging.
                record.msg = "Log message could not be encoded on this console."
                record.args = ()
        return True

class _JSONLineFormatter(logging.Formatter):
    """Emit structured JSON per line with UTF-8 (ensure_ascii=False)."""
    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat()
        payload = {
            "ts": ts,
            "level": record.levelname,
            "logger": record.name,
            "func": record.funcName,
            "line": record.lineno,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)

# --- Global Default Logger ---
default_logger = None

def get_default_logger():
    global default_logger
    if default_logger is None:
        default_logger = TDALogger(name="DefaultLogger", level="INFO")
    return default_logger

class TDALogger:
    """
    Advanced logging system for TDA experiments with engineering best practices.
    """

    def __init__(self,
                 name: str = "TDA_Pipeline",
                 log_dir: Union[str, Path] = "logs",
                 level: str = "INFO",
                 enable_console: bool = True,
                 enable_file: bool = True,
                 enable_json: bool = True):
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        self.logger.handlers.clear()

        text_fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%H:%M:%S"
        )

        if enable_console:
            console_handler = logging.StreamHandler(_utf8_stream(sys.stdout))
            console_handler.setLevel(getattr(logging, level.upper()))
            console_handler.setFormatter(text_fmt)
            console_handler.addFilter(_ConsoleUnicodeFilter())  # sanitize only for console
            self.logger.addHandler(console_handler)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if enable_file:
            log_file = self.log_dir / f"{self.name}_{timestamp}.log"
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setLevel(getattr(logging, level.upper()))
            file_handler.setFormatter(logging.Formatter(
                "%(asctime)s | %(levelname)-7s | %(name)s | %(funcName)s:%(lineno)d | %(message)s"
            ))
            self.logger.addHandler(file_handler)

        if enable_json:
            self.json_file = self.log_dir / f"{self.name}_{timestamp}.jsonl"
            json_handler = logging.FileHandler(self.json_file, encoding="utf-8")
            json_handler.setLevel(getattr(logging, level.upper()))
            json_handler.setFormatter(_JSONLineFormatter())
            self.logger.addHandler(json_handler)

        self.experiment_id = None
        self.experiment_start_time = None
        self.performance_data = {}

    def start_experiment(self, experiment_name: str, parameters: Dict[str, Any], **kwargs):
        self.experiment_id = str(uuid.uuid4())
        self.experiment_start_time = time.time()
        self.info(f"🚀 Starting experiment: {experiment_name} (ID: {self.experiment_id})")
        # pretty JSON parameters (will be sanitized for console if needed)
        self.info(f"Parameters: {json.dumps(parameters, indent=2, ensure_ascii=False)}")

    def end_experiment(self, status: str = "completed"):
        if not self.experiment_id:
            self.warning("No active experiment to end.")
            return
        duration = time.time() - self.experiment_start_time
        self.info(f"🏁 Experiment {self.experiment_id} finished with status: {status} in {duration:.2f}s.")
        self.experiment_id = None

    @contextmanager
    def log_performance(self, operation_name: str):
        start_time = time.time()
        start_mem = psutil.virtual_memory().used
        self.debug(f"Starting operation: {operation_name}")
        try:
            yield
        finally:
            duration = time.time() - start_time
            end_mem = psutil.virtual_memory().used
            mem_delta = end_mem - start_mem
            self.debug(f"Finished operation: {operation_name} in {duration:.2f}s. Memory delta: {mem_delta / 1e6:.2f} MB")

    def log_tda_results(self, **kwargs):
        self.info(f"TDA Results: {kwargs}")

    # These forwarders intentionally do NOT strip emojis here,
    # so file/JSON handlers keep full Unicode.
    def debug(self, message):
        self.logger.debug(message)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message, exc_info=False):
        self.logger.error(message, exc_info=exc_info)

def log_method_call(func):
    """Decorator for automatic method call logging with timing & exceptions."""
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        logger = getattr(self, 'logger', get_default_logger())
        func_name = func.__name__
        with logger.log_performance(func_name):
            try:
                return func(self, *args, **kwargs)
            except Exception as e:
                logger.error(f"Error in {func_name}: {e}", exc_info=True)
                raise
    return wrapper
