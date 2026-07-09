"""Moteur NER chirurgical pour manuscrits arabes — v3.0.

Multi-passes avec priorité :
  1. Dictionnaires (confiance 100)
  2. Structures syntaxiques (confiance 75-92)
  3. Entités temporelles (confiance 80-95)
  4. Siècles (confiance 90)
"""
import re
import hashlib
import logging
import time
import unicodedata
from typing import List, Dict, Any, Optional, Tuple

import pyarabic.araby as araby
from lxml import etree

import dictionnaires_patrimoine as dp
from config import config

logging.basicConfig(
    level=getattr(logging, config.log_level),
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("PatrimoineNER")

# Regex chirurgicales compilées une fois
TASHKEEL_REGEX = re.compile(
    r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06DC\u06DF-\u06E4\u06E7\u06E8\u06EA-\u06ED]"
)
TATWEEL_REGEX = re.compile(r"[\u0640]")
INVISIBLE_CHARS = re.compile(r"[\u200B-\u200F\u202A-\u202E\u2066-\u2069\uFEFF]")
NON_ARABIC_PUNCT = re.compile(
    r"[^\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\s0-9.,؛:!؟\-\"'«»()\n/]"
)
MULTI_SPACE = re.compile(r"\s+")


class MoteurExtraction:
    """Moteur d'extraction d'entités nommées — précision chirurgicale."""

    def __init__(self):
        self.intervals: List[Tuple[int, int]] = []
        self.stats: Dict[str, int] = {
            "textes_traites": 0,
            "entites_extraites": 0,
        }
        logger.info(f"Moteur NER v{dp.VERSION_DICT} | {dp.DICT_COUNT} entrées chargées")

    # =================================================================
    # NETTOYAGE CHIRURGICAL
    # =================================================================

    def nettoyer_texte_arabe(self, texte_brut: str) -> str:
        """Nettoyage chirurgical — préserve l'intégrité des noms propres."""
        if not texte_brut or not texte_brut.strip():
            return ""

        # 1. Unicode NFC
        texte = unicodedata.normalize("NFC", texte_brut)

        # 2. Caractères invisibles
        texte = INVISIBLE_CHARS.sub("", texte)

        # 3. Tashkeel (chirurgical — ne touche pas aux lettres)
        texte = TASHKEEL_REGEX.sub("", texte)

        # 4. Tatweel (kashida)
        texte = TATWEEL_REGEX.sub("", texte)

        # 5. Ligature
        texte = araby.normalize_ligature(texte)

        # ATTENTION : pas de normalize_hamza / normalize_alef (destructeur)

        # 6. Corrections OCR
        for pattern, correction in zip(dp.ERREURS_OCR, dp.CORRECTIONS_OCR):
            texte = pattern.sub(correction, texte)

        # 7. Ponctuation non arabe
        texte = NON_ARABIC_PUNCT.sub("", texte)

        # 8. Espaces
        texte = MULTI_SPACE.sub(" ", texte).strip()

        return texte

    # =================================================================
    # EXTRACTION MULTI-PASSES
    # =================================================================

    def extraire_entites(self, texte: str) -> List[Dict[str, Any]]:
        if not texte:
            return []

        self.intervals = []
        entites: List[Dict[str, Any]] = []

        # Passe 1 : Dictionnaires (priorité maximale)
        entites.extend(self._passe_dictionnaire(texte, dp.LIEUX_HISTORIQUES, "Lieu"))
        entites.extend(self._passe_dictionnaire(texte, dp.PERSONNAGES_HISTORIQUES, "Personnalité"))
        entites.extend(self._passe_dictionnaire(texte, dp.DYNASTIES, "Dynastie"))
        entites.extend(self._passe_dictionnaire(texte, dp.OEUVRES, "Œuvre"))
        entites.extend(self._passe_dictionnaire(texte, dp.TERMES_RELIGIEUX, "Terme religieux"))

        # Passe 2 : Structures syntaxiques
        entites.extend(self._passe_structurelle(texte))

        # Passe 3 : Temporel
        entites.extend(self._passe_temporelle(texte))
        entites.extend(self._passe_siecles(texte))

        # Post-traitement
        for item in entites:
            self._ajouter_contexte(texte, item)

        entites.sort(key=lambda x: x["position"])

        self.stats["textes_traites"] += 1
        self.stats["entites_extraites"] += len(entites)
        logger.info(f"Extraction: {len(entites)} entités trouvées")

        return entites

    def _passe_dictionnaire(self, texte: str, dico: Dict, type_entite: str) -> List[Dict]:
        results = []
        for nom, (sous_type, alias) in dico.items():
            cibles = [nom] + list(alias)
            for cible in cibles:
                if len(cible) < config.min_entity_length:
                    continue
                pattern = re.compile(
                    dp.ARABIC_WORD_BOUNDARY_LEFT
                    + re.escape(cible)
                    + dp.ARABIC_WORD_BOUNDARY_RIGHT
                )
                for match in pattern.finditer(texte):
                    s, e = match.start(), match.end()
                    if not self._chevauche(s, e):
                        results.append({
                            "type": type_entite,
                            "sous_type": sous_type,
                            "entite": nom,
                            "matched_text": match.group(),
                            "position": s,
                            "end": e,
                            "confiance": 100,
                            "source": "dictionary",
                            "method": "exact",
                        })
                        self._ajouter_interval(s, e)
        return results

    def _passe_structurelle(self, texte: str) -> List[Dict]:
        results = []
        for pattern in dp.STRUCTURES_NOMS:
            for match in pattern.finditer(texte):
                s, e = match.start(), match.end()
                if not self._chevauche(s, e):
                    matched = match.group().strip()
                    if "الشيخ" in matched or "الإمام" in matched or "سيدي" in matched:
                        confiance = 92
                    elif "بن" in matched:
                        confiance = 85
                    else:
                        confiance = 75
                    results.append({
                        "type": "Personnalité",
                        "sous_type": "Inconnu",
                        "entite": matched,
                        "matched_text": matched,
                        "position": s,
                        "end": e,
                        "confiance": confiance,
                        "source": "pattern",
                        "method": "structural",
                    })
                    self._ajouter_interval(s, e)
        return results

    def _passe_temporelle(self, texte: str) -> List[Dict]:
        results = []
        for match in dp.PATTERN_TEMPOREL.finditer(texte):
            s, e = match.start(), match.end()
            if not self._chevauche(s, e):
                matched = match.group().strip()
                if "هـ" in matched or "هجر" in matched:
                    sous_type, confiance = "Calendrier hégirien", 95
                elif "ميلاد" in matched:
                    sous_type, confiance = "Calendrier grégorien", 95
                else:
                    sous_type, confiance = "Date non spécifiée", 80
                results.append({
                    "type": "Date",
                    "sous_type": sous_type,
                    "entite": matched,
                    "matched_text": matched,
                    "position": s,
                    "end": e,
                    "confiance": confiance,
                    "source": "pattern",
                    "method": "temporal",
                })
                self._ajouter_interval(s, e)
        return results

    def _passe_siecles(self, texte: str) -> List[Dict]:
        results = []
        for match in dp.PATTERN_SIECLE.finditer(texte):
            s, e = match.start(), match.end()
            if not self._chevauche(s, e):
                results.append({
                    "type": "Date",
                    "sous_type": "Référence de siècle",
                    "entite": match.group().strip(),
                    "matched_text": match.group().strip(),
                    "position": s,
                    "end": e,
                    "confiance": 90,
                    "source": "pattern",
                    "method": "century",
                })
                self._ajouter_interval(s, e)
        return results

    # =================================================================
    # UTILITAIRES
    # =================================================================

    def _chevauche(self, start: int, end: int) -> bool:
        return any(start < s_end and end > s_start for s_start, s_end in self.intervals)

    def _ajouter_interval(self, start: int, end: int):
        self.intervals.append((start, end))

    def _ajouter_contexte(self, texte: str, item: Dict):
        w = config.context_window
        s_ctx = max(0, item["position"] - w)
        e_ctx = min(len(texte), item["end"] + w)
        item["contexte"] = texte[s_ctx:e_ctx]

    # =================================================================
    # HASH & TEI-XML
    # =================================================================

    @staticmethod
    def generer_hash_sha256(texte: str) -> str:
        return hashlib.sha256(texte.encode("utf-8")).hexdigest()

    @staticmethod
    def generer_tei_xml(
        texte: str,
        entites: List[Dict[str, Any]],
        metadata: Optional[Dict] = None,
    ) -> str:
        """Génère un document TEI-XML conforme avec header complet."""
        nsmap = {"tei": "http://www.tei-c.org/ns/1.0"}
        TEI = "{%s}" % nsmap["tei"]

        root = etree.Element(TEI + "TEI", nsmap=nsmap)

        # --- Header ---
        header = etree.SubElement(root, TEI + "teiHeader")
        file_desc = etree.SubElement(header, TEI + "fileDesc")
        title_stmt = etree.SubElement(file_desc, TEI + "titleStmt")
        etree.SubElement(title_stmt, TEI + "title").text = (
            metadata.get("title", "Manuscrit Indexé") if metadata else "Manuscrit Indexé"
        )
        etree.SubElement(title_stmt, TEI + "author").text = "TechCulture AI Studio"

        pub_stmt = etree.SubElement(file_desc, TEI + "publicationStmt")
        etree.SubElement(pub_stmt, TEI + "publisher").text = "TechCulture AI Studio"
        etree.SubElement(pub_stmt, TEI + "date").text = time.strftime("%Y-%m-%d")

        source_desc = etree.SubElement(file_desc, TEI + "sourceDesc")
        ms_desc = etree.SubElement(source_desc, TEI + "msDesc")
        ms_id = etree.SubElement(ms_desc, TEI + "msIdentifier")
        etree.SubElement(ms_id, TEI + "idno").text = (
            metadata.get("hash", "unknown") if metadata else "unknown"
        )

        profile_desc = etree.SubElement(header, TEI + "profileDesc")
        lang_usage = etree.SubElement(profile_desc, TEI + "langUsage")
        lang = etree.SubElement(lang_usage, TEI + "language")
        lang.set("ident", "ar")
        lang.text = "Arabic"

        # --- Corps ---
        text_node = etree.SubElement(root, TEI + "text")
        body = etree.SubElement(text_node, TEI + "body")

        tag_map = {
            "Personnalité": "persName",
            "Lieu": "placeName",
            "Date": "date",
            "Dynastie": "orgName",
            "Œuvre": "title",
            "Terme religieux": "rs",
        }

        last_node = body
        last_pos = 0

        for item in entites:
            s = item["position"]
            e = item.get("end", s + len(item["entite"]))
            txt_before = texte[last_pos:s]

            if last_node is body:
                last_node.text = (last_node.text or "") + txt_before
            else:
                last_node.tail = (last_node.tail or "") + txt_before

            elem = etree.SubElement(body, TEI + tag_map.get(item["type"], "rs"))
            elem.set("type", item.get("sous_type", "unknown"))
            cert = "high" if item["confiance"] >= 90 else "medium" if item["confiance"] >= 70 else "low"
            elem.set("cert", cert)
            elem.set("source", item.get("source", "unknown"))
            elem.text = item["entite"]

            last_node = elem
            last_pos = e

        txt_after = texte[last_pos:]
        if last_node is body:
            last_node.text = (last_node.text or "") + txt_after
        else:
            last_node.tail = (last_node.tail or "") + txt_after

        # --- Back : Index ---
        back = etree.SubElement(text_node, TEI + "back")

        persons = [e for e in entites if e["type"] == "Personnalité"]
        if persons:
            list_p = etree.SubElement(back, TEI + "listPerson")
            for p in sorted(set(p["entite"] for p in persons)):
                pe = etree.SubElement(list_p, TEI + "person")
                etree.SubElement(pe, TEI + "persName").text = p

        places = [e for e in entites if e["type"] == "Lieu"]
        if places:
            list_pl = etree.SubElement(back, TEI + "listPlace")
            for p in sorted(set(p["entite"] for p in places)):
                pe = etree.SubElement(list_pl, TEI + "place")
                etree.SubElement(pe, TEI + "placeName").text = p

        return etree.tostring(
            root, pretty_print=True, xml_declaration=True, encoding="UTF-8"
        ).decode("utf-8")

    # =================================================================
    # PIPELINE COMPLET
    # =================================================================

    def analyser_complete(self, texte: str) -> Dict[str, Any]:
        """Pipeline complet avec métadonnées riches."""
        t0 = time.perf_counter()

        texte_nettoye = self.nettoyer_texte_arabe(texte)
        entites = self.extraire_entites(texte_nettoye)
        hash_doc = self.generer_hash_sha256(texte)
        xml_output = self.generer_tei_xml(
            texte_nettoye, entites, metadata={"hash": hash_doc, "title": "Manuscrit TechCulture"}
        )

        elapsed = time.perf_counter() - t0
        stats_ent = {}
        for e in entites:
            stats_ent[e["type"]] = stats_ent.get(e["type"], 0) + 1

        return {
            "texte_original": texte,
            "texte_nettoye": texte_nettoye,
            "entites": entites,
            "hash": hash_doc,
            "xml": xml_output,
            "metadata": {
                "version_moteur": dp.VERSION_DICT,
                "nombre_entites": len(entites),
                "temps_traitement": round(elapsed, 3),
                "statistiques": stats_ent,
                "mots_total": len(texte_nettoye.split()),
            },
        }

    def get_stats(self) -> Dict[str, Any]:
        return self.stats.copy()
