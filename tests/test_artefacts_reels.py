"""Vérifie les vrais artefacts du dépôt (ignoré s'ils sont absents)."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import api.main as api_main
from fraude_xai.modele import ModeleFraude

ART = Path(__file__).resolve().parent.parent / "artefacts"
pytestmark = pytest.mark.skipif(not (ART / "modele_xgb.json").exists(), reason="artefacts réels absents")


def test_exemple_api_reproduit_le_score_du_notebook(monkeypatch):
    monkeypatch.setattr(api_main, "_modele", ModeleFraude.charger(ART))
    ex = json.loads((ART / "exemple_api.json").read_text(encoding="utf-8"))
    r = TestClient(api_main.app).post("/explain", json=ex["demande"])
    assert r.status_code == 200, r.text
    r = r.json()
    assert r["score"] == pytest.approx(ex["score_attendu"], abs=1e-6)
    assert r["alerte"] and len(r["codes_motifs"]) >= 3
