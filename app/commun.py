"""Fonctions partagées par les pages de l'application (chargement des artefacts en cache)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE))

from fraude_xai.modele import ModeleFraude  # noqa: E402

ARTEFACTS = Path(os.environ.get("DOSSIER_ARTEFACTS", RACINE / "artefacts"))
MODELES = ["XGBoost", "Régression logistique", "Isolation Forest", "Autoencodeur"]
COULEURS = {"XGBoost": "#c0392b", "Régression logistique": "#2e86c1", "Isolation Forest": "#27ae60",
            "Autoencodeur": "#8e44ad", "LOF": "#d68910"}
FRAUDE, NORMAL = "#c0392b", "#5d6d7e"


def configurer(titre: str, icone: str = "💳"):
    st.set_page_config(page_title=f"{titre} – Fraude explicable", page_icon=icone, layout="wide")
    if not (ARTEFACTS / "config.json").exists():
        st.error(f"Artefacts introuvables dans `{ARTEFACTS}`. Exécute le notebook 04 et copie le dossier "
                 "`artefacts/` à la racine du dépôt.")
        st.stop()


@st.cache_data
def config() -> dict:
    return json.loads((ARTEFACTS / "config.json").read_text(encoding="utf-8"))


@st.cache_data
def eda() -> dict:
    return json.loads((ARTEFACTS / "eda.json").read_text(encoding="utf-8"))


@st.cache_data
def scores() -> pd.DataFrame:
    return pd.read_parquet(ARTEFACTS / "scores.parquet")


@st.cache_data
def transactions() -> pd.DataFrame:
    return pd.read_parquet(ARTEFACTS / "transactions_app.parquet")


@st.cache_data
def tableau(nom: str, **kw) -> pd.DataFrame | None:
    chemin = ARTEFACTS / nom
    return pd.read_csv(chemin, **kw) if chemin.exists() else None


@st.cache_resource
def modele() -> ModeleFraude:
    return ModeleFraude.charger(ARTEFACTS)


def nombre(x: float) -> str:
    return f"{x:,.0f}".replace(",", " ")


def dollars(x: float) -> str:
    return f"{nombre(x)} $"


def pourcent(x: float, d: int = 1) -> str:
    return f"{x * 100:.{d}f} %"
