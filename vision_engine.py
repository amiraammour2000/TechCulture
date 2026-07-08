"""Moteur Vision OCR — Prétraitement chirurgical + PaddleOCR.

Pipeline :
  1. Conversion bytes → image
  2. Niveaux de gris + upscale
  3. Débruitage (NLMeans)
  4. Correction d'inclinaison (deskew)
  5. Binarisation (Sauvola / Otsu / Adaptive / Niblack)
  6. Amélioration contraste (CLAHE)
  7. PaddleOCR
  8. Post-correction
"""
import cv2
import numpy as np
import logging
import warnings
import time
import re
from typing import Dict, Any

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger("VisionEngine")


class VisionEngine:
    """Moteur de reconnaissance optique de manuscrits arabes."""

    def __init__(self, binarization_method: str = "sauvola"):
        self._ocr = None
        self.binarization_method = binarization_method
        self.stats = {"images_traitees": 0, "lignes_extraites": 0}

    def _get_ocr_model(self):
        if self._ocr is None:
            logger.info("Chargement PaddleOCR (Arabe)...")
            from paddleocr import PaddleOCR

            self._ocr = PaddleOCR(lang="ar")
            logger.info("✅ PaddleOCR chargé")
        return self._ocr

    def _bytes_to_image(self, image_bytes: bytes) -> np.ndarray:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Image corrompue ou format non supporté")
        return img

    # =================================================================
    # PRÉTRAITEMENT CHIRURGICAL
    # =================================================================

    def pretraiter_manuscrit(self, img: np.ndarray) -> np.ndarray:
        # 1. Gris
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()

        # 2. Upscale si petite
        h, w = gray.shape
        if max(h, w) < 1000:
            gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        # 3. Débruitage NLMeans (préserve les contours)
        gray = cv2.fastNlMeansDenoising(gray, h=10)

        # 4. Deskew
        gray = self._deskew(gray)

        # 5. Binarisation
        binary = self._binarize(gray)

        # 6. CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        binary = clahe.apply(binary)

        return binary

    def _deskew(self, img: np.ndarray) -> np.ndarray:
        thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        coords = np.column_stack(np.where(thresh > 0))
        if len(coords) < 100:
            return img

        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        if abs(angle) < 0.5 or abs(angle) > 30:
            return img

        h, w = img.shape
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)

    def _binarize(self, img: np.ndarray) -> np.ndarray:
        method = self.binarization_method.lower()

        if method == "otsu":
            _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif method == "adaptive":
            binary = cv2.adaptiveThreshold(
                img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15
            )
        elif method == "sauvola":
            binary = self._sauvola(img, window=25, k=0.2, r=128)
        elif method == "niblack":
            binary = self._niblack(img, window=25, k=-0.2)
        else:
            _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return binary

    def _sauvola(self, img: np.ndarray, window: int = 25, k: float = 0.2, r: float = 128) -> np.ndarray:
        """Binarisation de Sauvola — optimale pour manuscrits."""
        f = img.astype(np.float32)
        mean = cv2.boxFilter(f, cv2.CV_32F, (window, window), normalize=True)
        mean_sq = cv2.boxFilter(f * f, cv2.CV_32F, (window, window), normalize=True)
        std = np.sqrt(np.maximum(mean_sq - mean * mean, 0))
        threshold = mean * (1 + k * (std / r - 1))
        return np.where(f > threshold, 255, 0).astype(np.uint8)

    def _niblack(self, img: np.ndarray, window: int = 25, k: float = -0.2) -> np.ndarray:
        f = img.astype(np.float32)
        mean = cv2.boxFilter(f, cv2.CV_32F, (window, window), normalize=True)
        mean_sq = cv2.boxFilter(f * f, cv2.CV_32F, (window, window), normalize=True)
        std = np.sqrt(np.maximum(mean_sq - mean * mean, 0))
        threshold = mean + k * std
        return np.where(f > threshold, 255, 0).astype(np.uint8)

    # =================================================================
    # POST-CORRECTION
    # =================================================================

    def _post_correct(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\s+([.,؛:!؟])", r"\1", text)
        text = re.sub(r"([.,؛:!؟])([^\s])", r"\1 \2", text)
        return text.strip()

    # =================================================================
    # EXTRACTION PRINCIPALE
    # =================================================================

    def extraire_texte(self, image_bytes: bytes, use_preprocessing: bool = True) -> str:
        t0 = time.perf_counter()
        logger.info("Début extraction OCR...")

        img = self._bytes_to_image(image_bytes)
        logger.info(f"Image: {img.shape[1]}x{img.shape[0]}")

        img_to_feed = self.pretraiter_manuscrit(img) if use_preprocessing else img

        ocr_model = self._get_ocr_model()
        resultats = ocr_model.ocr(img_to_feed)

        lignes = []
        scores = []

        if resultats and len(resultats) > 0 and resultats[0]:
            for ligne in resultats[0]:
                try:
                    if isinstance(ligne, dict):
                        text = ligne.get("rec_text", ligne.get("text", ""))
                        score = ligne.get("score", ligne.get("rec_score", 1.0))
                    elif isinstance(ligne, (list, tuple)) and len(ligne) >= 2:
                        text = ligne[1][0] if isinstance(ligne[1], (list, tuple)) else str(ligne[1])
                        score = ligne[1][1] if isinstance(ligne[1], (list, tuple)) else 1.0
                    else:
                        continue

                    if score > 0.5 and text.strip():
                        lignes.append(text.strip())
                        scores.append(float(score))
                except Exception as e:
                    logger.warning(f"Erreur ligne OCR: {e}")

        texte = self._post_correct("\n".join(lignes))

        elapsed = time.perf_counter() - t0
        self.stats["images_traitees"] += 1
        self.stats["lignes_extraites"] += len(lignes)
        avg = sum(scores) / len(scores) if scores else 0

        logger.info(f"OCR: {len(lignes)} lignes | confiance {avg:.2%} | {elapsed:.2f}s")
        return texte

    def get_stats(self) -> Dict[str, Any]:
        return self.stats.copy()