"""Explications (Module 5) : contributions SHAP et codes motifs.

Les contributions sont calculées par XGBoost lui-même (``pred_contribs=True``) : ce sont les valeurs
SHAP exactes de TreeSHAP, en log-odds, y compris pour la variable catégorielle du groupe MCC.
Les deux variables d'heure (sin et cos) sont regroupées en une seule, « heure ».
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

# Libellés courts des variables (graphiques)
LIBELLES = {
    "heure": "heure", "jour_semaine": "jour de la semaine", "week_end": "week-end", "nuit": "nuit (0 h – 6 h)",
    "montant": "montant", "log_montant": "montant (log)", "remboursement": "remboursement",
    "en_ligne": "paiement en ligne", "bande": "bande magnétique", "etranger": "à l'étranger",
    "err_cvv": "erreur CVV", "err_expiration": "erreur date d'expiration", "err_numero": "erreur numéro de carte",
    "err_pin": "erreur code PIN", "err_solde": "solde insuffisant", "err_technique": "incident technique",
    "err_code_postal": "erreur code postal", "une_erreur": "au moins une erreur",
    "h_depuis_prec": "temps depuis la transaction précédente", "nb_tx_1h": "transactions sur 1 h",
    "nb_tx_24h": "transactions sur 24 h", "nb_tx_30j": "transactions sur 30 jours", "nb_err_24h": "erreurs sur 24 h",
    "moy_30j": "montant moyen sur 30 jours", "ratio_montant_30j": "montant / moyenne 30 jours",
    "nb_etranger_24h": "transactions à l'étranger sur 24 h", "nb_nouveaux_marchands_24h": "nouveaux marchands sur 24 h",
    "montant_24h": "montant dépensé sur 24 h", "nouveau_marchand": "nouveau marchand",
    "nouvelle_categorie": "nouvelle catégorie", "nouvelle_ville": "nouvelle ville",
    "part_en_ligne_passee": "habitude d'achat en ligne", "mcc_groupe": "catégorie de marchand",
    "mcc_risque": "catégorie de marchand",
}

# Catégories de marchand (MCC) les plus courantes, pour des codes motifs lisibles
MCC_LIBELLES = {
    "aerien": "compagnie aérienne", "hotel": "hôtel", "location_voiture": "location de voiture",
    "4112": "train", "4121": "taxi / VTC", "4411": "croisière", "4511": "transport aérien", "4784": "péage",
    "4814": "télécommunications", "4829": "virement d'argent", "4899": "télévision par câble",
    "4900": "services publics", "5045": "ordinateurs et périphériques", "5094": "bijouterie",
    "5192": "livres et journaux", "5193": "fleuriste (gros)", "5211": "bricolage", "5251": "quincaillerie",
    "5300": "grande surface de gros", "5310": "magasin discount", "5311": "grand magasin", "5411": "supermarché",
    "5499": "épicerie fine", "5541": "station-service", "5651": "vêtements", "5655": "articles de sport",
    "5661": "chaussures", "5712": "meubles", "5719": "décoration", "5722": "électroménager",
    "5732": "électronique", "5733": "instruments de musique", "5812": "restaurant", "5813": "bar",
    "5814": "restauration rapide", "5815": "contenus numériques", "5816": "jeux en ligne",
    "5912": "pharmacie", "5921": "vins et spiritueux", "5932": "antiquités", "5942": "librairie",
    "5947": "cadeaux et souvenirs", "5970": "loisirs créatifs", "5977": "cosmétiques", "6300": "assurance",
    "7011": "hébergement", "7210": "blanchisserie", "7349": "nettoyage", "7531": "carrosserie",
    "7538": "garage automobile", "7542": "lavage auto", "7801": "jeux d'argent", "7802": "courses hippiques",
    "7832": "cinéma", "7996": "parc d'attractions", "8011": "médecin", "8021": "dentiste",
    "8041": "chiropracteur", "8043": "opticien", "8049": "podologue", "8062": "hôpital",
    "8099": "services médicaux", "8111": "services juridiques", "8931": "comptabilité", "9402": "poste",
}


def contributions(booster: xgb.Booster, X: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Calcule les contributions SHAP (log-odds) par variable, heure regroupée, et la valeur de base."""
    d = xgb.DMatrix(X, enable_categorical=True)
    c = booster.predict(d, pred_contribs=True)
    contrib = pd.DataFrame(c[:, :-1], columns=X.columns, index=X.index)
    base = c[:, -1]
    if {"heure_sin", "heure_cos"} <= set(contrib.columns):
        contrib.insert(0, "heure", contrib.pop("heure_sin") + contrib.pop("heure_cos"))
    return contrib, base


def _heure(row) -> str:
    angle = np.arctan2(row["heure_sin"], row["heure_cos"])
    h = (angle / (2 * np.pi) * 24) % 24
    return f"{int(h):02d} h {int(round((h % 1) * 60)) % 60:02d}"


def phrase(variable: str, row: pd.Series) -> str:
    """Renvoie une phrase lisible par un analyste pour une variable et sa valeur sur la transaction."""
    v = row.get(variable, np.nan)
    if variable == "heure":
        return f"Heure de la transaction : {_heure(row)}"
    if variable in ("mcc_groupe", "mcc_risque"):
        code = str(row.get("mcc_groupe", ""))
        nom = MCC_LIBELLES.get(code)
        return f"Catégorie de marchand : {nom} (MCC {code})" if nom and code.isdigit() else \
            f"Catégorie de marchand : {nom or code}"
    if variable in ("montant", "log_montant"):
        return f"Montant de {row['montant']:,.2f} $".replace(",", " ")
    if variable == "ratio_montant_30j":
        if pd.isna(v):
            return "Aucun historique de montant sur 30 jours"
        return f"Montant égal à {v:.1f} fois la moyenne des 30 derniers jours"
    if variable == "moy_30j":
        return "Pas d'historique sur 30 jours" if pd.isna(v) else f"Montant moyen sur 30 jours : {v:,.0f} $"
    if variable == "h_depuis_prec":
        if pd.isna(v):
            return "Première transaction connue de la carte"
        return f"{v * 60:.0f} minutes depuis la transaction précédente" if v < 1 else \
            f"{v:.1f} h depuis la transaction précédente"
    if variable == "montant_24h":
        return f"{v:,.0f} $ dépensés sur les 24 dernières heures".replace(",", " ")
    if variable == "part_en_ligne_passee":
        return "Pas d'historique" if pd.isna(v) else f"{v * 100:.0f} % d'achats en ligne dans l'historique de la carte"
    compteurs = {"nb_tx_1h": "transaction(s) dans l'heure précédente", "nb_tx_24h": "transaction(s) sur 24 h",
                 "nb_tx_30j": "transaction(s) sur 30 jours", "nb_err_24h": "erreur(s) sur 24 h",
                 "nb_etranger_24h": "transaction(s) à l'étranger sur 24 h",
                 "nb_nouveaux_marchands_24h": "nouveau(x) marchand(s) sur 24 h"}
    if variable in compteurs:
        return f"{int(v)} {compteurs[variable]}"
    binaires = {
        "etranger": ("Transaction à l'étranger", "Transaction aux États-Unis"),
        "en_ligne": ("Paiement en ligne", "Paiement en magasin"),
        "bande": ("Paiement par bande magnétique", "Paiement sans bande magnétique"),
        "nouveau_marchand": ("Premier achat de la carte chez ce marchand", "Marchand habituel de la carte"),
        "nouvelle_categorie": ("Première fois dans cette catégorie de marchand", "Catégorie habituelle"),
        "nouvelle_ville": ("Première transaction de la carte dans cette ville", "Ville habituelle"),
        "nuit": ("Transaction de nuit (0 h – 6 h)", "Transaction en journée"),
        "week_end": ("Transaction le week-end", "Transaction en semaine"),
        "remboursement": ("Montant négatif (remboursement)", "Débit"),
        "err_cvv": ("Erreur de CVV", "Pas d'erreur de CVV"),
        "err_expiration": ("Erreur de date d'expiration", "Pas d'erreur de date d'expiration"),
        "err_numero": ("Erreur de numéro de carte", "Pas d'erreur de numéro"),
        "err_pin": ("Erreur de code PIN", "Pas d'erreur de code PIN"),
        "err_solde": ("Solde insuffisant", "Solde suffisant"),
        "err_technique": ("Incident technique", "Pas d'incident technique"),
        "err_code_postal": ("Erreur de code postal", "Pas d'erreur de code postal"),
        "une_erreur": ("Au moins une erreur à l'autorisation", "Aucune erreur à l'autorisation"),
    }
    if variable in binaires:
        return binaires[variable][0] if v == 1 else binaires[variable][1]
    if variable == "jour_semaine":
        jours = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
        return f"Transaction un {jours[int(v)]}"
    return f"{LIBELLES.get(variable, variable)} = {v}"


def codes_motifs(contrib: pd.Series, row: pd.Series, n: int = 5, minimum: float = 0.05) -> list[dict]:
    """Renvoie les n variables qui ont le plus poussé vers l'alerte (contributions positives)."""
    pos = contrib[contrib > minimum].sort_values(ascending=False).head(n)
    return [{"variable": var, "libellé": LIBELLES.get(var, var), "contribution": round(float(c), 4),
             "motif": phrase(var, row)} for var, c in pos.items()]


def contributions_principales(contrib: pd.Series, row: pd.Series, n: int = 10) -> list[dict]:
    """Renvoie les n contributions les plus fortes en valeur absolue, dans les deux sens."""
    top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(n)
    return [{"variable": var, "libellé": LIBELLES.get(var, var), "contribution": round(float(c), 4),
             "sens": "vers la fraude" if c > 0 else "vers le normal", "valeur": phrase(var, row)}
            for var, c in top.items()]
