"""Chargement du modèle retenu et notation de transactions (utilisé par l'API et l'application)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from .explications import codes_motifs, contributions, contributions_principales
from .variables import COLONNES_BRUTES, construire_variables, matrice

DOSSIER_ARTEFACTS = Path(__file__).resolve().parent.parent / "artefacts"


def corriger_proba(ps: np.ndarray, beta: float) -> np.ndarray:
    """Ramène à l'échelle réelle une probabilité prédite après rééchantillonnage : p = beta*ps / (beta*ps - ps + 1)."""
    return beta * ps / (beta * ps - ps + 1)


@dataclass
class ModeleFraude:
    booster: xgb.Booster
    config: dict

    @classmethod
    def charger(cls, dossier: Path | str = DOSSIER_ARTEFACTS) -> "ModeleFraude":
        """Charge le modèle XGBoost et sa configuration depuis le dossier des artefacts."""
        dossier = Path(dossier)
        booster = xgb.Booster()
        booster.load_model(dossier / "modele_xgb.json")
        config = json.loads((dossier / "config.json").read_text(encoding="utf-8"))
        return cls(booster, config)

    @property
    def seuil(self) -> float:
        return float(self.config["seuil"])

    def scorer(self, variables: pd.DataFrame) -> np.ndarray:
        """Renvoie la probabilité de fraude (échelle réelle) à partir des variables déjà construites."""
        X = matrice(variables, self.config)
        ps = self.booster.predict(xgb.DMatrix(X, enable_categorical=True))
        beta = float(self.config.get("BETA", 1.0))
        return corriger_proba(ps, beta) if beta != 1 else ps

    def expliquer(self, variables: pd.DataFrame, n_motifs: int = 5) -> list[dict]:
        """Renvoie le score, la décision et l'explication de chaque ligne de ``variables``."""
        X = matrice(variables, self.config)
        scores = self.scorer(variables)
        contrib, base = contributions(self.booster, X)
        sortie = []
        for i, (idx, row) in enumerate(variables.iterrows()):
            c = contrib.loc[idx]
            sortie.append({
                "score": float(scores[i]),
                "alerte": bool(scores[i] >= self.seuil),
                "seuil": self.seuil,
                "valeur_de_base_logodds": float(base[i]),
                "codes_motifs": codes_motifs(c, row, n=n_motifs),
                "contributions": contributions_principales(c, row),
            })
        return sortie

    def noter_transaction(self, historique: pd.DataFrame, transaction: pd.DataFrame) -> tuple[pd.Series, dict]:
        """Construit les variables de la transaction à partir de l'historique de sa carte, puis l'explique.

        L'historique doit couvrir au moins les 30 derniers jours pour des variables exactes.
        """
        brut = pd.concat([historique, transaction], ignore_index=True)[COLONNES_BRUTES]
        brut["_cible"] = np.r_[np.zeros(len(historique), dtype=bool), np.ones(len(transaction), dtype=bool)]
        brut["_ordre"] = np.arange(len(brut))
        var = construire_variables(brut)
        ligne = var[var["_cible"]].sort_values("_ordre").tail(1)
        return ligne.iloc[0], self.expliquer(ligne)[0]
