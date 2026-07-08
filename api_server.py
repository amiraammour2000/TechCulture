"""API REST FastAPI pour TechCulture AI Studio."""
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Optional
import logging

from engine import MoteurExtraction
from vision_engine import VisionEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("API")

app = FastAPI(title="TechCulture AI Studio API", version="3.0.0")
moteur_ner = MoteurExtraction()
moteur_vision = VisionEngine()


class TextInput(BaseModel):
    text: str
    generate_tei: bool = True


@app.get("/")
async def root():
    return {"service": "TechCulture AI Studio", "version": "3.0.0", "status": "ok"}


@app.post("/api/v1/analyze/text")
async def analyze_text(inp: TextInput):
    try:
        res = moteur_ner.analyser_complete(inp.text)
        return {
            "success": True,
            "hash": res["hash"],
            "texte_nettoye": res["texte_nettoye"],
            "entites": res["entites"],
            "metadata": res["metadata"],
            "xml": res["xml"] if inp.generate_tei else None,
        }
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/v1/analyze/image")
async def analyze_image(file: UploadFile = File(...), use_preprocessing: bool = True):
    try:
        img_bytes = await file.read()
        txt = moteur_vision.extraire_texte(img_bytes, use_preprocessing)
        if not txt.strip():
            raise HTTPException(422, "Aucun texte extrait")
        res = moteur_ner.analyser_complete(txt)
        return {
            "success": True,
            "texte_ocr": txt,
            "hash": res["hash"],
            "entites": res["entites"],
            "metadata": res["metadata"],
            "xml": res["xml"],
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/v1/stats")
async def stats():
    return {"ner": moteur_ner.get_stats(), "vision": moteur_vision.get_stats()}


@app.get("/api/v1/health")
async def health():
    return {"status": "healthy"}