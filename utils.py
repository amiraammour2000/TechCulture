"""Utilitaires partagés."""
import hashlib
import json
import time
import logging
from functools import wraps
from typing import Any, Optional

logger = logging.getLogger("TechCulture")


def generate_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info(f"[{func.__name__}] Exécuté en {elapsed:.3f}s")
        return result
    return wrapper


def validate_arabic_text(text: str) -> bool:
    if not text or not text.strip():
        return False
    arabic_chars = sum(1 for c in text if "\u0600" <= c <= "\u06FF")
    return arabic_chars >= 3


def safe_json_serialize(obj: Any) -> str:
    def default(o):
        if isinstance(o, set):
            return list(o)
        if hasattr(o, "__dict__"):
            return o.__dict__
        return str(o)
    return json.dumps(obj, ensure_ascii=False, indent=2, default=default)