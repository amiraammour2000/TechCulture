# app.py
import streamlit as st
import xml.dom.minidom
import json
import time
from typing import Dict, Any

from engine import MoteurExtraction
from vision_engine import VisionEngine
from dictionnaires_patrimoine import (
    VERSION_DICT, DICT_COUNT,
    LIEUX_HISTORIQUES, PERSONNAGES_HISTORIQUES, DYNASTIES, OEUVRES, TERMES_RELIGIEUX,
)

# =====================================================================
# CONFIG
# =====================================================================
st.set_page_config(page_title="TechCulture AI Studio Pro", page_icon="📜", layout="wide")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Amiri:wght@400;700&family=Cairo:wght@400;600;700&display=swap');
#MainMenu, footer, header {visibility: hidden;}
.block-container {padding-top: 1.5rem; max-width: 1400px;}
.main-title {
    text-align: center; background: linear-gradient(135deg, #1a1a2e, #16213e, #0f3460);
    color: white; padding: 2rem; border-radius: 15px; margin-bottom: 1.5rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.15);
}
.main-title h1 {
    font-family: 'Cairo', sans-serif; font-size: 2.5rem; margin: 0;
    background: linear-gradient(90deg, #e94560, #f5a623);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.main-title p {font-family: 'Amiri', serif; font-size: 1.1rem; opacity: 0.9; margin-top: 0.5rem;}
.entity-card {
    background: #f8f9fa; padding: 1rem; border-radius: 10px;
    border-left: 4px solid #0f3460; margin: 0.5rem 0; transition: transform 0.2s;
}
.entity-card:hover {transform: translateX(5px);}
.hl-person {border-left-color: #28a745;} .hl-place {border-left-color: #007bff;}
.hl-date {border-left-color: #fd7e14;} .hl-work {border-left-color: #6f42c1;}
.hl-dynasty {border-left-color: #dc3545;} .hl-religious {border-left-color: #20c997;}
.rtl-text {direction: rtl; text-align: right; font-family: 'Amiri', serif;}
.badge {display: inline-block; padding: 3px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: 600;}
.badge-high {background: #d4edda; color: #155724;}
.badge-medium {background: #fff3cd; color: #856404;}
.badge-low {background: #f8d7da; color: #721c24;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

st.markdown("""
<div class="main-title">
    <h1>📜 TechCulture AI Studio</h1>
    <p>Moteur d'Indexation Sémantique & OCR de Manuscrits Arabes<br>
    <small>-مركز البحث العلمي و التقني لتطوير اللغة العربية -تلمسان</small></p>
</div>
""", unsafe_allow_html=True)

# =====================================================================
# SIDEBAR
# =====================================================================
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.markdown("#### 🖼️ Vision")
    binarization = st.selectbox("Binarisation", ["sauvola", "otsu", "adaptive", "niblack"], index=0)
    st.markdown("#### 📊 Informations")
    st.info(f"**Version:** {VERSION_DICT}\n\n**Entrées:** {DICT_COUNT}\n\n**Moteur:** NER Multi-Passes v3.0")
    if st.button("🗑️ Réinitialiser"):
        st.session_state.clear()
        st.rerun()

# =====================================================================
# INIT MOTEURS
# =====================================================================
if "moteur_ner" not in st.session_state:
    st.session_state.moteur_ner = MoteurExtraction()
if "moteur_vision" not in st.session_state:
    st.session_state.moteur_vision = VisionEngine(binarization_method=binarization)

# =====================================================================
# FONCTIONS
# =====================================================================
ENTITY_META = {
    "Personnalité": ("🟢", "hl-person", "#28a745"),
    "Lieu": ("🔵", "hl-place", "#007bff"),
    "Date": ("🟠", "hl-date", "#fd7e14"),
    "Dynastie": ("🔴", "hl-dynasty", "#dc3545"),
    "Œuvre": ("🟣", "hl-work", "#6f42c1"),
    "Terme religieux": ("🟢", "hl-religious", "#20c997"),
}

def _badge(conf):
    if conf >= 90: return '<span class="badge badge-high">Élevée</span>'
    elif conf >= 70: return '<span class="badge badge-medium">Moyenne</span>'
    return '<span class="badge badge-low">Faible</span>'

def _highlight_text(texte, entites):
    if not entites:
        return f'<div class="rtl-text" style="padding:1rem;">{texte}</div>'
    parts = []
    last = 0
    for ent in sorted(entites, key=lambda x: x["position"]):
        s, e = ent["position"], ent.get("end", ent["position"] + len(ent["entite"]))
        parts.append(texte[last:s])
        _, _, color = ENTITY_META.get(ent["type"], ("", "", "#6c757d"))
        parts.append(
            f'<mark style="background:{color}22;border-bottom:2px solid {color};'
            f'padding:2px 4px;border-radius:3px;" '
            f'title="{ent["type"]} — {ent["sous_type"]} ({ent["confiance"]}%)">'
            f'{ent["entite"]}</mark>'
        )
        last = e
    parts.append(texte[last:])
    return f'<div class="rtl-text" style="padding:1rem;line-height:2;font-size:1.1rem;">{"".join(parts)}</div>'

def afficher_resultats(resultat):
    entites = resultat["entites"]
    texte = resultat["texte_nettoye"]
    h = resultat["hash"]
    xml_str = resultat["xml"]
    meta = resultat["metadata"]

    st.success(f"✅ Analyse terminée en {meta['temps_traitement']}s | {meta['nombre_entites']} entités")

    cols = st.columns(len(meta["statistiques"]) + 2)
    cols[0].metric("Entités", meta["nombre_entites"])
    cols[1].metric("Mots", meta["mots_total"])
    for i, (t, n) in enumerate(meta["statistiques"].items()):
        cols[i + 2].metric(t, n)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📝 Texte Indexé", "🔍 Entités", "📊 Stats", "🌳 TEI-XML", "📤 Export"])

    with tab1:
        st.markdown("#### Texte avec entités surlignées")
        st.markdown(_highlight_text(texte, entites), unsafe_allow_html=True)
        st.markdown("---")
        st.text_area("Texte nettoyé", texte, height=200, key=f"txt_{h[:8]}")
        st.caption(f"🔐 SHA-256 : `{h}`")

    with tab2:
        if not entites:
            st.info("Aucune entité.")
        else:
            c1, c2, c3 = st.columns(3)
            with c1:
                types = list(set(e["type"] for e in entites))
                sel = st.multiselect("Type", types, default=types, key=f"ft_{h[:8]}")
            with c2:
                mc = st.slider("Confiance min", 0, 100, 0, 10, key=f"fc_{h[:8]}")
            with c3:
                srch = st.text_input("Rechercher", "", key=f"fs_{h[:8]}")

            for ent in entites:
                if ent["type"] not in sel or ent["confiance"] < mc:
                    continue
                if srch and srch not in ent["entite"]:
                    continue
                icon, css, _ = ENTITY_META.get(ent["type"], ("⚪", "", ""))
                st.markdown(f"""
                <div class="entity-card {css}">
                    <div style="display:flex;justify-content:space-between;align-items:center;">
                        <div>{icon} <strong>{ent["entite"]}</strong>
                        <span style="color:#666;font-size:0.9rem;">({ent["type"]} — {ent["sous_type"]})</span>
                        {_badge(ent["confiance"])}</div>
                        <div style="font-size:0.85rem;color:#999;">{ent["confiance"]}% | {ent.get("source","")}</div>
                    </div>
                    <div style="margin-top:0.5rem;font-size:0.85rem;color:#555;direction:rtl;text-align:right;font-family:'Amiri',serif;">…{ent.get("contexte","")}…</div>
                </div>
                """, unsafe_allow_html=True)

    with tab3:
        import plotly.express as px
        import pandas as pd
        c1, c2 = st.columns(2)
        with c1:
            df = pd.DataFrame(meta["statistiques"].items(), columns=["Type", "Nombre"])
            st.plotly_chart(px.pie(df, values="Nombre", names="Type", title="Distribution"), use_container_width=True)
        with c2:
            confs = [e["confiance"] for e in entites]
            if confs:
                st.plotly_chart(px.histogram(x=confs, nbins=10, title="Confiances", labels={"x": "%"}), use_container_width=True)

    with tab4:
        st.code(xml.dom.minidom.parseString(xml_str).toprettyxml(indent="  "), language="xml")

    with tab5:
        c1, c2, c3, c4 = st.columns(4)
        c1.download_button("🌳 XML", xml_str, file_name=f"manuscrit_{h[:8]}.xml", mime="application/xml")
        c2.download_button("📋 JSON", json.dumps({"hash": h, "texte": texte, "entites": entites, "metadata": meta}, ensure_ascii=False, indent=2), file_name=f"manuscrit_{h[:8]}.json", mime="application/json")
        c3.download_button("📄 TXT", "\n".join(f"[{e['type']}] {e['entite']} ({e['confiance']}%)" for e in entites), file_name=f"entites_{h[:8]}.txt")
        import csv, io
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=["type", "sous_type", "entite", "confiance", "source", "contexte"])
        w.writeheader()
        for e in entites:
            w.writerow({k: e.get(k, "") for k in ["type", "sous_type", "entite", "confiance", "source", "contexte"]})
        c4.download_button("📊 CSV", buf.getvalue().encode("utf-8"), file_name=f"entites_{h[:8]}.csv")

# =====================================================================
# ONGLETS
# =====================================================================
tabs = st.tabs(["📝 Mode Texte", "📜 Mode Vision", "📚 Documentation"])

# --- Texte ---
with tabs[0]:
    st.markdown("### Saisie directe de texte arabe")
    with st.expander("💡 Exemples"):
        exemples = [
            "وُلد الشيخ سيدي محمد بن عبد الرحمان في ورقلة سنة 1180 هجري، وتوفي في غرداية عام 1250 هـ. درس على يد أبي العباس أحمد وقرأ صحيح البخاري وموطأ الإمام مالك.",
            "رحل ابن خلدون إلى القاهرة في القرن الثامن هجري، حيث التقى بعلماء الأزهر. ألف كتاب العبر وديوان المبتدأ والخبر.",
            "كانت زاوية كنتة مركزاً علمياً في الصحراء الجزائرية، يتردد عليها طلاب العلم من تقرت والمنيعة.",
        ]
        for i, ex in enumerate(exemples):
            if st.button(f"Exemple {i+1}", key=f"ex_{i}"):
                st.session_state["zone_saisie"] = ex
                st.rerun()

    texte_brut = st.text_area("Collez votre texte :", height=250, placeholder="بسم الله الرحمن الرحيم...", key="zone_saisie")

    if st.button("🚀 Analyser", type="primary", use_container_width=True):
        if texte_brut.strip():
            with st.spinner("🔬 Pipeline chirurgical..."):
                res = st.session_state.moteur_ner.analyser_complete(texte_brut)
                afficher_resultats(res)
        else:
            st.warning("⚠️ Veuillez saisir du texte.")

# --- Vision ---
with tabs[1]:
    st.markdown("### OCR de Manuscrits")
    c1, c2 = st.columns([1, 2])
    with c1:
        fup = st.file_uploader("Image", type=["jpg", "jpeg", "png", "tiff", "bmp", "webp"], key="upl")
        pre = st.checkbox("Prétraitement chirurgical", value=True)
    with c2:
        if fup:
            st.image(fup, width="stretch", caption="Aperçu")
            st.info(f"📁 {fup.name} | {len(fup.getvalue())/1024:.1f} KB")

    if st.button("🔬 Lancer OCR + NER", type="primary", use_container_width=True, disabled=not fup):
        try:
            with st.spinner("🧠 Étape 1/2 : OCR..."):
                st.session_state.moteur_vision.binarization_method = binarization
                txt_ocr = st.session_state.moteur_vision.extraire_texte(fup.getvalue(), use_preprocessing=pre)
            if not txt_ocr.strip():
                st.error("❌ Aucun texte extrait.")
            else:
                st.info(f"📄 {len(txt_ocr)} caractères extraits")
                with st.spinner("🔬 Étape 2/2 : NER..."):
                    res = st.session_state.moteur_ner.analyser_complete(txt_ocr)
                    afficher_resultats(res)
        except Exception as e:
            st.error(f"❌ {e}")

# --- Doc ---
with tabs[2]:
    st.markdown(f"""
    ## 📚 Documentation

    | Catégorie | Entrées |
    |-----------|---------|
    | Lieux | {len(LIEUX_HISTORIQUES)} |
    | Personnages | {len(PERSONNAGES_HISTORIQUES)} |
    | Dynasties | {len(DYNASTIES)} |
    | Œuvres | {len(OEUVRES)} |
    | Termes religieux | {len(TERMES_RELIGIEUX)} |
    | **Total** | **{DICT_COUNT}** |

    ### Binarisation
    | Méthode | Usage |
    |---------|-------|
    | Sauvola | ⭐ Manuscrits anciens |
    | Otsu | Documents modernes |
    | Adaptive | Éclairage inégal |
    | Niblack | Manuscrits dégradés |
    """)
