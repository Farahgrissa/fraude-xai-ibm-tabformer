"""API de notation et d'explication des transactions (Module 6).

Routes :
- ``GET  /health``               : état du service ;
- ``GET  /modele``               : informations sur le modèle, le seuil et ses performances ;
- ``POST /score``                : transaction brute + historique de la carte -> score et décision ;
- ``POST /explain``              : idem, avec codes motifs et contributions SHAP ;
- ``POST /score/variables``      : notation directe à partir de variables déjà construites.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from fraude_xai import __version__
from fraude_xai.modele import DOSSIER_ARTEFACTS, ModeleFraude
from fraude_xai.variables import CATEGORIELLE, VARIABLES

app = FastAPI(
    title="API de détection de fraude explicable",
    description="Notation de transactions par carte bancaire (IBM TabFormer) : score, alerte selon un seuil "
                "choisi par le coût, et codes motifs SHAP pour l'analyste.",
    version=__version__,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_modele: ModeleFraude | None = None


def modele() -> ModeleFraude:
    global _modele
    if _modele is None:
        dossier = os.environ.get("DOSSIER_ARTEFACTS", str(DOSSIER_ARTEFACTS))
        try:
            _modele = ModeleFraude.charger(dossier)
        except FileNotFoundError as e:
            raise HTTPException(503, f"Artefacts du modèle introuvables ({e.filename}).") from e
    return _modele


class Transaction(BaseModel):
    date: datetime = Field(..., examples=["2019-06-14T02:41:00"])
    montant: float = Field(..., description="Montant en dollars (négatif = remboursement)", examples=[849.0])
    canal: Literal["Chip Transaction", "Swipe Transaction", "Online Transaction"] = "Chip Transaction"
    marchand: str = Field(..., description="Identifiant du marchand", examples=["7148423482893829373"])
    ville: str | None = Field(None, examples=["Rome"])
    etat: str | None = Field(None, description="État américain (2 lettres) ou pays à l'étranger",
                             examples=["Italy"])
    mcc: int = Field(..., description="Code de catégorie de marchand", examples=[5094])
    erreurs: str | None = Field(None, description="Erreurs séparées par des virgules", examples=["Bad CVV"])


class Demande(BaseModel):
    carte: str = Field("0_0", description="Identifiant de carte (utilisé seulement pour regrouper)")
    historique: list[Transaction] = Field(
        default_factory=list,
        description="Historique complet de la carte (les variables de nouveauté et d'habitude en dépendent ; "
                    "au minimum les 30 derniers jours)")
    transaction: Transaction


class Variables(BaseModel):
    variables: dict[str, float | int | str | None] = Field(
        ..., description=f"Valeurs des {len(VARIABLES)} variables et de '{CATEGORIELLE}'")


def _brut(transactions: list[Transaction], carte: str) -> pd.DataFrame:
    user, _, card = carte.partition("_")
    lignes = [{
        "User": user or "0", "Card": card or "0", "date": pd.Timestamp(t.date).tz_localize(None),
        "Amount": t.montant, "Use Chip": t.canal, "Merchant Name": t.marchand,
        "Merchant City": t.ville if t.ville is not None else ("ONLINE" if t.canal.startswith("Online") else None),
        "Merchant State": t.etat, "MCC": t.mcc, "Errors?": t.erreurs,
    } for t in transactions]
    df = pd.DataFrame(lignes, columns=["User", "Card", "date", "Amount", "Use Chip", "Merchant Name",
                                       "Merchant City", "Merchant State", "MCC", "Errors?"])
    df["date"] = pd.to_datetime(df["date"])
    return df


def _noter(d: Demande):
    m = modele()
    ligne, resultat = m.noter_transaction(_brut(d.historique, d.carte), _brut([d.transaction], d.carte))
    resultat["variables"] = {k: (None if pd.isna(ligne[k]) else (ligne[k].item() if hasattr(ligne[k], "item")
                                                                  else ligne[k]))
                             for k in VARIABLES + [CATEGORIELLE]}
    return resultat


@app.get("/health")
def health():
    try:
        m = modele()
        return {"statut": "ok", "version": __version__, "seuil": m.seuil}
    except HTTPException as e:
        return {"statut": "dégradé", "détail": e.detail}


@app.get("/modele")
def infos_modele():
    c = modele().config
    return {k: c.get(k) for k in ["modele", "STRATEGIE", "seuil", "cout_alerte", "BUDGET_JOUR",
                                  "metriques_validation", "metriques_test", "periode", "version_donnees"]}


@app.post("/score")
def score(d: Demande):
    r = _noter(d)
    return {"score": r["score"], "alerte": r["alerte"], "seuil": r["seuil"]}


@app.post("/explain")
def explain(d: Demande):
    return _noter(d)


@app.post("/score/variables")
def score_variables(v: Variables):
    manquantes = [k for k in VARIABLES + [CATEGORIELLE] if k not in v.variables]
    if manquantes:
        raise HTTPException(422, f"Variables manquantes : {manquantes}")
    df = pd.DataFrame([v.variables])
    return modele().expliquer(df)[0]
