"""Données simulées au format IBM TabFormer et artefacts d'un petit modèle, pour les tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from fraude_xai.variables import CATEGORIELLE, VARIABLES, construire_variables, matrice


def transactions(n_cartes: int = 12, seed: int = 0) -> pd.DataFrame:
    """Génère des transactions au format TabFormer, avec une campagne de fraude sur une carte sur deux."""
    rng = np.random.default_rng(seed)
    lignes = []
    for c in range(n_cartes):
        n = int(rng.integers(150, 300))
        dates = pd.Timestamp("2018-01-01") + pd.to_timedelta(np.sort(rng.uniform(0, 400 * 24, n)), unit="h")
        fraude = np.zeros(n, dtype=int)
        if c % 2 == 0:
            d = int(rng.integers(20, n - 8))
            fraude[d:d + 5] = 1
        for i in range(n):
            f = fraude[i] == 1
            online = rng.random() < (0.6 if f else 0.1)
            lignes.append({
                "User": c, "Card": 0, "date": dates[i].floor("min"),
                "Amount": float(np.round(rng.lognormal(3.4 + 1.2 * f, 0.8), 2)),
                "Use Chip": "Online Transaction" if online else ("Chip Transaction" if rng.random() < .8
                                                                 else "Swipe Transaction"),
                "Merchant Name": int(rng.integers(1000, 10 ** 9)) if f else int(rng.integers(0, 15)),
                "Merchant City": "ONLINE" if online else ("Rome" if f else rng.choice(["Austin", "Dallas"])),
                "Merchant State": None if online else ("Italy" if f else "TX"),
                "MCC": int(rng.choice([5094, 5732]) if f else rng.choice([5411, 5812, 3006])),
                "Errors?": "Bad CVV" if (f and rng.random() < .2) else None,
                "fraude": int(f),
            })
    return pd.DataFrame(lignes)


def creer_artefacts(dossier: Path, utilise_num: bool = False, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Entraîne un petit XGBoost et écrit modele_xgb.json et config.json, comme le notebook 04."""
    brut = transactions(seed=seed)
    var = construire_variables(brut)
    y = var["fraude"].to_numpy()
    taux = float(y.mean())
    stat = var.groupby(CATEGORIELLE)["fraude"].agg(["sum", "count"])
    config = {
        "VARIABLES": VARIABLES, "CATEGORIELLE": CATEGORIELLE, "UTILISE_NUM": utilise_num,
        "categories_mcc": sorted(var[CATEGORIELLE].unique()),
        "mcc_risque": ((stat["sum"] + 10 * taux) / (stat["count"] + 10)).to_dict(),
        "taux_global": taux, "medianes": var[VARIABLES].median().to_dict(), "BETA": 0.5,
        "STRATEGIE": {"nom": "test"}, "seuil": 0.2, "cout_alerte": 5.0, "BUDGET_JOUR": 10,
    }
    X = matrice(var, config)
    m = xgb.XGBClassifier(n_estimators=30, max_depth=3, enable_categorical=True, tree_method="hist",
                          random_state=seed).fit(X, y)
    dossier.mkdir(parents=True, exist_ok=True)
    m.get_booster().save_model(dossier / "modele_xgb.json")
    (dossier / "config.json").write_text(json.dumps(config, default=float), encoding="utf-8")
    return brut, var
