"""Tests unitaires du moteur NER."""
import pytest
from engine import MoteurExtraction


@pytest.fixture
def moteur():
    return MoteurExtraction()


def test_nettoyage_tashkeel(moteur):
    texte = "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ"
    r = moteur.nettoyer_texte_arabe(texte)
    assert "بسم الله" in r


def test_extraction_lieu(moteur):
    e = moteur.extraire_entites("وُلد في ورقلة سنة 1180 هجري")
    assert any(x["type"] == "Lieu" and "ورقلة" in x["entite"] for x in e)


def test_extraction_personnage(moteur):
    e = moteur.extraire_entites("قال ابن خلدون في كتابه")
    assert any(x["type"] == "Personnalité" for x in e)


def test_extraction_date(moteur):
    e = moteur.extraire_entites("في سنة 1180 هجري حدث ذلك")
    assert any(x["type"] == "Date" for x in e)


def test_tei_xml(moteur):
    txt = "ورقلة وغرداية"
    e = moteur.extraire_entites(txt)
    xml = moteur.generer_tei_xml(txt, e)
    assert "<TEI" in xml and "placeName" in xml


def test_hash(moteur):
    h1 = moteur.generer_hash_sha256("test")
    assert h1 == moteur.generer_hash_sha256("test")
    assert h1 != moteur.generer_hash_sha256("diff")


def test_no_overlap(moteur):
    e = moteur.extraire_entites("ورقلة")
    intervals = [(x["position"], x.get("end", x["position"] + len(x["entite"]))) for x in e]
    for i, (s1, e1) in enumerate(intervals):
        for j, (s2, e2) in enumerate(intervals):
            if i != j:
                assert not (s1 < e2 and e1 > s2)