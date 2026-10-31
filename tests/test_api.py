import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import api.main as api_main
from fraude_xai.modele import ModeleFraude
from fraude_xai.variables import CATEGORIELLE, VARIABLES


def vers_api(r):
    na = lambda v: v is None or (isinstance(v, float) and np.isnan(v))  # noqa: E731
    return {"date": r["date"].isoformat(), "montant": float(r["Amount"]), "canal": r["Use Chip"],
            "marchand": str(r["Merchant Name"]), "ville": None if na(r["Merchant City"]) else r["Merchant City"],
            "etat": None if na(r["Merchant State"]) else r["Merchant State"], "mcc": int(r["MCC"]),
            "erreurs": None if na(r["Errors?"]) else r["Errors?"]}


@pytest.fixture
def client(artefacts, monkeypatch):
    dossier, brut, var = artefacts
    monkeypatch.setattr(api_main, "_modele", ModeleFraude.charger(dossier))
    return TestClient(api_main.app), brut, var


def demande(brut):
    carte = brut[brut["fraude"] == 1]["User"].iloc[0]
    c = brut[brut["User"] == carte].sort_values("date").reset_index(drop=True)
    i = int(c.index[c["fraude"] == 1][0])
    return {"carte": f"{carte}_0", "historique": [vers_api(r) for _, r in c.iloc[:i].iterrows()],
            "transaction": vers_api(c.iloc[i])}, c.iloc[i]


def test_health(client):
    c, _, _ = client
    r = c.get("/health").json()
    assert r["statut"] == "ok" and "seuil" in r


def test_explain_reproduit_le_calcul_hors_ligne(client):
    c, brut, var = client
    d, cible = demande(brut)
    r = c.post("/explain", json=d)
    assert r.status_code == 200, r.text
    r = r.json()
    ref = var[(var["carte"] == d["carte"]) & (var["date"] == cible["date"])].head(1)
    attendu = api_main._modele.scorer(ref)[0]
    assert r["score"] == pytest.approx(float(attendu), abs=1e-6)
    assert isinstance(r["codes_motifs"], list) and set(VARIABLES) <= set(r["variables"])


def test_score(client):
    c, brut, _ = client
    d, _ = demande(brut)
    r = c.post("/score", json=d).json()
    assert set(r) == {"score", "alerte", "seuil"}


def test_score_variables(client):
    c, _, var = client
    ligne = var.iloc[50]
    v = {k: (None if pd.isna(ligne[k]) else (ligne[k].item() if hasattr(ligne[k], "item") else ligne[k]))
         for k in VARIABLES + [CATEGORIELLE]}
    r = c.post("/score/variables", json={"variables": v})
    assert r.status_code == 200, r.text
    assert r.json()["score"] == pytest.approx(float(api_main._modele.scorer(var.iloc[[50]])[0]), abs=1e-6)
    assert c.post("/score/variables", json={"variables": {"montant": 3}}).status_code == 422
