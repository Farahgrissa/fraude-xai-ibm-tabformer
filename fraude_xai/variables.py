"""Construction des variables (Module 2) et encodage pour les modèles.

Règle : chaque variable d'une transaction n'utilise que le passé de sa carte. Les fenêtres glissantes
utilisent ``closed="left"``, qui exclut la transaction en cours.

Le même code sert au notebook, à l'application et à l'API : une transaction notée par l'API reçoit
exactement les variables vues pendant l'entraînement.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Colonnes brutes attendues (format IBM TabFormer, après lecture et création de "date")
COLONNES_BRUTES = ["User", "Card", "date", "Amount", "Use Chip", "Merchant Name", "Merchant City",
                   "Merchant State", "MCC", "Errors?"]

VARIABLES = [
    "heure_sin", "heure_cos", "jour_semaine", "week_end", "nuit",
    "montant", "log_montant", "remboursement", "en_ligne", "bande", "etranger",
    "err_cvv", "err_expiration", "err_numero", "err_pin", "err_solde", "err_technique", "err_code_postal", "une_erreur",
    "h_depuis_prec", "nb_tx_1h", "nb_tx_24h", "nb_tx_30j", "nb_err_24h", "moy_30j", "ratio_montant_30j",
    "nb_etranger_24h", "nb_nouveaux_marchands_24h", "montant_24h",
    "nouveau_marchand", "nouvelle_categorie", "nouvelle_ville", "part_en_ligne_passee",
]
CATEGORIELLE = "mcc_groupe"

ERREURS = {"err_cvv": "Bad CVV", "err_expiration": "Bad Expiration", "err_numero": "Bad Card Number",
           "err_pin": "Bad PIN", "err_solde": "Insufficient Balance", "err_technique": "Technical Glitch",
           "err_code_postal": "Bad Zipcode"}


def groupe_mcc(mcc: pd.Series) -> np.ndarray:
    """Regroupe les codes MCC de voyage (une compagnie aérienne ou un hôtel par code)."""
    mcc = mcc.astype(int)
    return np.select([mcc.between(3000, 3299), mcc.between(3351, 3441), mcc.between(3501, 3999)],
                     ["aerien", "location_voiture", "hotel"], default=mcc.astype(str))


def construire_variables(ech: pd.DataFrame) -> pd.DataFrame:
    """Calcule toutes les variables. ``ech`` contient les colonnes de ``COLONNES_BRUTES``."""
    data = ech.drop_duplicates().copy()
    data["carte"] = data["User"].astype(str) + "_" + data["Card"].astype(str)
    data = data.sort_values(["carte", "date"], kind="mergesort").reset_index(drop=True)

    # Variables temporelles
    h = data["date"].dt.hour + data["date"].dt.minute / 60
    data["heure_sin"] = np.sin(2 * np.pi * h / 24)
    data["heure_cos"] = np.cos(2 * np.pi * h / 24)
    data["jour_semaine"] = data["date"].dt.dayofweek.astype("int8")
    data["week_end"] = (data["jour_semaine"] >= 5).astype("int8")
    data["nuit"] = (h < 6).astype("int8")

    # Variables de la transaction
    data["montant"] = data["Amount"].astype("float32")
    data["remboursement"] = (data["montant"] < 0).astype("int8")
    data["log_montant"] = np.sign(data["montant"]) * np.log1p(data["montant"].abs())
    canal = data["Use Chip"].astype(str)
    data["en_ligne"] = canal.str.contains("Online").astype("int8")
    data["bande"] = canal.str.contains("Swipe").astype("int8")
    # À l'étranger, "Merchant State" contient le nom du pays (plus de 2 lettres). Une valeur manquante
    # est lue "nan" comme pendant l'entraînement, pour garder exactement la même définition.
    etat = data["Merchant State"].astype(object).where(data["Merchant State"].notna(), "nan").astype(str)
    data["etranger"] = ((etat.str.len() > 2) & (data["en_ligne"] == 0)).astype("int8")
    err = data["Errors?"].astype(str)
    for nom, motif in ERREURS.items():
        data[nom] = err.str.contains(motif, regex=False).astype("int8")
    data["une_erreur"] = data[list(ERREURS)].max(axis=1).astype("int8")
    data["mcc_groupe"] = groupe_mcc(data["MCC"])

    # Variables de nouveauté : premier achat de la carte chez ce marchand, dans cette catégorie, dans cette ville
    data["nouveau_marchand"] = (data.groupby(["carte", "Merchant Name"]).cumcount() == 0).astype("int8")
    data["nouvelle_categorie"] = (data.groupby(["carte", "mcc_groupe"]).cumcount() == 0).astype("int8")
    data["nouvelle_ville"] = ((data.groupby(["carte", "Merchant City"], observed=True).cumcount() == 0)
                              & (data["en_ligne"] == 0)).astype("int8")

    # Variables comportementales : fenêtres glissantes sur le passé de la carte
    g = data.groupby("carte", sort=False)
    data["h_depuis_prec"] = (g["date"].diff().dt.total_seconds() / 3600).astype("float32")
    data["part_en_ligne_passee"] = (g["en_ligne"].cumsum() - data["en_ligne"]) / g.cumcount().replace(0, np.nan)
    data["montant_pos"] = data["montant"].clip(lower=0)

    roul = data.set_index("date").groupby("carte", sort=False)

    def fenetre(col, taille, fonction):
        return getattr(roul[col].rolling(taille, closed="left"), fonction)().to_numpy()

    data["nb_tx_1h"] = np.nan_to_num(fenetre("montant", "1h", "count")).astype("int16")
    data["nb_tx_24h"] = np.nan_to_num(fenetre("montant", "24h", "count")).astype("int16")
    data["nb_tx_30j"] = np.nan_to_num(fenetre("montant", "30D", "count")).astype("int16")
    data["nb_err_24h"] = np.nan_to_num(fenetre("une_erreur", "24h", "sum")).astype("int16")
    data["nb_etranger_24h"] = np.nan_to_num(fenetre("etranger", "24h", "sum")).astype("int16")
    data["nb_nouveaux_marchands_24h"] = np.nan_to_num(fenetre("nouveau_marchand", "24h", "sum")).astype("int16")
    data["montant_24h"] = np.nan_to_num(fenetre("montant_pos", "24h", "sum")).astype("float32")
    data["moy_30j"] = fenetre("montant", "30D", "mean").astype("float32")
    data["ratio_montant_30j"] = (data["montant"] / (data["moy_30j"].abs() + 1)).astype("float32")
    return data


def matrice(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Construit la matrice d'entrée de XGBoost selon l'encodage retenu à l'entraînement (``config``).

    - ``UTILISE_NUM`` faux : groupe MCC en catégorie native ;
    - ``UTILISE_NUM`` vrai (stratégie SMOTE) : valeurs manquantes remplacées par les médianes de
      l'entraînement et groupe MCC remplacé par son taux de fraude lissé (``mcc_risque``).
    """
    if config.get("UTILISE_NUM"):
        X = df[VARIABLES].fillna(pd.Series(config["medianes"])).astype("float32")
        X["mcc_risque"] = (df[CATEGORIELLE].astype(str).map(config["mcc_risque"])
                           .fillna(config["taux_global"]).astype("float32"))
        return X
    X = df[VARIABLES].astype("float32")
    X[CATEGORIELLE] = pd.Categorical(df[CATEGORIELLE].astype(str), categories=config["categories_mcc"])
    return X
