"""Évaluation sous budget et choix du seuil par le coût (Module 4).

Coût d'un seuil τ : montants des fraudes manquées + coût d'analyse × nombre d'alertes.
Seule la partie positive des montants compte (124 fraudes ont un montant négatif).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def precision_rappel_k(y: np.ndarray, score: np.ndarray, k: int) -> tuple[float, float]:
    """Calcule la précision et le rappel parmi les k transactions les plus suspectes."""
    k = int(min(k, len(y)))
    ordre = np.argsort(-score, kind="stable")[:k]
    vp = y[ordre].sum()
    return vp / k, vp / max(y.sum(), 1)


def metriques_classement(y, score, k) -> dict:
    """Calcule la PR-AUC, la ROC-AUC, la précision@k et le rappel@k."""
    p, r = precision_rappel_k(y, score, k)
    return {"PR-AUC": average_precision_score(y, score), "ROC-AUC": roc_auc_score(y, score),
            "précision@k": p, "rappel@k": r}


def metriques_seuil(y, score, montant, seuil, cout_alerte) -> dict:
    """Calcule les métriques et le coût au seuil donné (alerte si score >= seuil)."""
    alerte = score >= seuil
    vp = int((alerte & (y == 1)).sum())
    fp = int((alerte & (y == 0)).sum())
    fn = int((~alerte & (y == 1)).sum())
    vn = int((~alerte & (y == 0)).sum())
    precision = vp / (vp + fp) if vp + fp else 0.0
    rappel = vp / (vp + fn) if vp + fn else 0.0
    f2 = 5 * precision * rappel / (4 * precision + rappel) if precision + rappel else 0.0
    denom = np.sqrt(float(vp + fp) * (vp + fn) * (vn + fp) * (vn + fn))
    mcc = (vp * vn - fp * fn) / denom if denom else 0.0
    m = np.clip(montant, 0, None)
    perte = float(m[(~alerte) & (y == 1)].sum())
    analyse = cout_alerte * (vp + fp)
    return {"seuil": float(seuil), "alertes": vp + fp, "VP": vp, "FP": fp, "FN": fn, "VN": vn,
            "précision": precision, "rappel": rappel, "F2": f2, "MCC": mcc,
            "fraudes manquées ($)": perte, "coût d'analyse ($)": analyse, "coût total ($)": perte + analyse,
            "montant fraudé total ($)": float(m[y == 1].sum())}


def courbe_cout(y, score, montant, cout_alerte, n_points: int = 400) -> pd.DataFrame:
    """Calcule le coût total sur une grille de seuils (quantiles du score), de façon vectorisée."""
    y = np.asarray(y)
    score = np.asarray(score, dtype=float)
    m = np.clip(np.asarray(montant, dtype=float), 0, None)
    ordre = np.argsort(-score, kind="stable")
    s, yy, mm = score[ordre], y[ordre], m[ordre]
    # Seuils candidats : scores distincts les plus élevés + quantiles, pour couvrir les petits budgets
    qs = np.unique(np.concatenate([np.quantile(score, np.linspace(0.5, 1, n_points)),
                                   s[: min(len(s), 5000)][:: max(1, min(len(s), 5000) // n_points)]]))
    n_alertes = np.searchsorted(-s, -qs, side="right")  # Nombre de scores >= seuil
    vp_cum = np.concatenate([[0], np.cumsum(yy)])
    detecte_cum = np.concatenate([[0], np.cumsum(mm * yy)])
    vp = vp_cum[n_alertes]
    detecte = detecte_cum[n_alertes]
    perte = mm[yy == 1].sum() - detecte
    df = pd.DataFrame({"seuil": qs, "alertes": n_alertes, "VP": vp, "fraudes manquées ($)": perte,
                       "coût d'analyse ($)": cout_alerte * n_alertes})
    df["coût total ($)"] = df["fraudes manquées ($)"] + df["coût d'analyse ($)"]
    df["précision"] = np.where(df["alertes"] > 0, df["VP"] / df["alertes"].clip(lower=1), 0)
    df["rappel"] = df["VP"] / max(y.sum(), 1)
    return df.sort_values("seuil").reset_index(drop=True)


def seuil_optimal(y, score, montant, cout_alerte) -> float:
    """Renvoie le seuil qui minimise le coût total."""
    c = courbe_cout(y, score, montant, cout_alerte)
    return float(c.loc[c["coût total ($)"].idxmin(), "seuil"])


def seuil_pour_budget(score, k: int) -> float:
    """Renvoie le seuil qui produit environ k alertes (score de la k-ième transaction la plus suspecte)."""
    k = int(min(max(k, 1), len(score)))
    return float(np.sort(np.asarray(score))[::-1][k - 1])
