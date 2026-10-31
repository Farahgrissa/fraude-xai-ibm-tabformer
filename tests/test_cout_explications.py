import numpy as np
import xgboost as xgb

from fraude_xai.cout import courbe_cout, metriques_seuil, precision_rappel_k, seuil_optimal, seuil_pour_budget
from fraude_xai.explications import codes_motifs, contributions
from fraude_xai.modele import ModeleFraude, corriger_proba
from fraude_xai.variables import matrice


def donnees(seed=0, n=5000):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.02).astype(int)
    s = np.clip(rng.normal(0.2 + 0.5 * y, 0.15), 0, 1)
    m = rng.lognormal(3.5, 1, n) * np.where(rng.random(n) < 0.03, -1, 1)
    return y, s, m


def test_metriques_seuil_coherentes():
    y, s, m = donnees()
    r = metriques_seuil(y, s, m, 0.5, 5.0)
    assert r["VP"] + r["FN"] == y.sum()
    assert r["coût total ($)"] == r["fraudes manquées ($)"] + 5.0 * r["alertes"]
    assert r["fraudes manquées ($)"] >= 0  # Montants négatifs ignorés
    assert -1 <= r["MCC"] <= 1 and 0 <= r["F2"] <= 1


def test_courbe_cout_egale_calcul_direct():
    y, s, m = donnees(1)
    c = courbe_cout(y, s, m, 5.0)
    for _, ligne in c.sample(20, random_state=0).iterrows():
        r = metriques_seuil(y, s, m, ligne["seuil"], 5.0)
        assert r["alertes"] == ligne["alertes"]
        assert abs(r["coût total ($)"] - ligne["coût total ($)"]) < 1e-6 * max(1, r["coût total ($)"])


def test_seuil_optimal_minimise_le_cout():
    y, s, m = donnees(2)
    t = seuil_optimal(y, s, m, 5.0)
    best = metriques_seuil(y, s, m, t, 5.0)["coût total ($)"]
    for autre in np.quantile(s, [0.5, 0.8, 0.95, 0.99]):
        assert best <= metriques_seuil(y, s, m, autre, 5.0)["coût total ($)"] + 1e-6


def test_budget_et_precision_k():
    y, s, _ = donnees(3)
    t = seuil_pour_budget(s, 100)
    assert (s >= t).sum() == 100
    p, r = precision_rappel_k(y, s, 100)
    assert 0 <= p <= 1 and 0 <= r <= 1


def test_correction_proba():
    ps = np.array([0.1, 0.5, 0.9])
    assert np.allclose(corriger_proba(ps, 1.0), ps)
    assert (corriger_proba(ps, 0.1) < ps).all()


def test_contributions_additives(artefacts):
    dossier, _, var = artefacts
    m = ModeleFraude.charger(dossier)
    X = matrice(var.head(200), m.config)
    C, base = contributions(m.booster, X)
    marge = m.booster.predict(xgb.DMatrix(X, enable_categorical=True), output_margin=True)
    assert np.allclose(C.sum(axis=1).to_numpy() + base, marge, atol=1e-4)
    assert "heure" in C.columns and "heure_sin" not in C.columns


def test_codes_motifs(artefacts):
    dossier, _, var = artefacts
    m = ModeleFraude.charger(dossier)
    ligne = var[var["fraude"] == 1].head(1)
    res = m.expliquer(ligne)[0]
    assert 0 <= res["score"] <= 1
    assert len(res["codes_motifs"]) <= 5
    assert all(cm["contribution"] > 0 and isinstance(cm["motif"], str) for cm in res["codes_motifs"])
    C, _ = contributions(m.booster, matrice(ligne, m.config))
    assert codes_motifs(C.iloc[0], ligne.iloc[0]) == res["codes_motifs"]
