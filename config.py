"""Configuration centrale — TechCulture AI Studio v3.0."""
from pathlib import Path
from dataclasses import dataclass

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
LOG_DIR = BASE_DIR / "logs"
CACHE_DIR = BASE_DIR / "cache"

for d in [DATA_DIR, LOG_DIR, CACHE_DIR]:
    d.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class AppConfig:
    app_name: str = "TechCulture AI Studio"
    app_version: str = "3.0.0-PROFESSIONAL"

    # OCR / Vision
    ocr_lang: str = "ar"
    ocr_confidence_threshold: float = 0.5
    enable_deskew: bool = True
    enable_denoise: bool = True
    binarization_method: str = "sauvola"

    # NLP
    enable_fuzzy_matching: bool = True
    fuzzy_threshold: int = 85
    context_window: int = 50
    min_entity_length: int = 2

    # TEI
    tei_encoding: str = "UTF-8"
    tei_pretty_print: bool = True

    # Logging
    log_level: str = "INFO"


config = AppConfig()