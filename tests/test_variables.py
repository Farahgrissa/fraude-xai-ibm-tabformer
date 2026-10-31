import numpy as np
import pandas as pd

from fraude_xai.variables import VARIABLES, construire_variables
from tests.donnees_simulees import transactions

COMPORTEMENT = ["nb_tx_1h", "nb_tx_24h", "nb_tx_30j", "nb_err_24h", "nb_etranger_24h", "nb_nouveaux_marchands_24h",
                "montant_24h", "moy_30j", "h_depuis_prec", "nouveau_marchand", "nouvelle_categorie",
                "nouvelle_ville", "part_en_ligne_passee"]


def test_toutes_les_variables_existent():
    var = construire_variables(transactions(n_cartes=3))
    assert set(VARIABLES) <= set(var.columns)


def test_pas_de_fuite_en_retirant_le_futur():
    brut = transactions(n_cartes=4, seed=1)
    complet = construire_variables(brut)
    coupe = brut["date"].quantile(0.5)
    partiel = construire_variables(brut[brut["date"] <= coupe])
    a = complet[complet["date"] <= coupe].sort_values(["carte", "date"]).reset_index(drop=True)
    b = partiel.sort_values(["carte", "date"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(a[COMPORTEMENT].astype(float), b[COMPORTEMENT].astype(float), atol=1e-4)


def test_fenetre_exclut_la_transaction_en_cours():
    t0 = pd.Timestamp("2019-01-01 10:00")
    dates = [t0, t0 + pd.Timedelta(minutes=10), t0 + pd.Timedelta(minutes=20)]
    brut = pd.DataFrame({"User": 1, "Card": 0, "date": dates,
                         "Amount": [10.0, 20.0, 600.0], "Use Chip": "Chip Transaction", "Merchant Name": [1, 1, 2],
                         "Merchant City": "Austin", "Merchant State": "TX", "MCC": 5411, "Errors?": None})
    v = construire_variables(brut)
    assert v["nb_tx_1h"].tolist() == [0, 1, 2]
    assert v["nouveau_marchand"].tolist() == [1, 0, 1]
    assert np.isnan(v["moy_30j"].iloc[0]) and v["moy_30j"].iloc[2] == 15.0
    assert v["montant_24h"].iloc[2] == 30.0


def test_etranger_et_canal():
    v = construire_variables(transactions(n_cartes=4))
    assert (v.loc[v["en_ligne"] == 1, "etranger"] == 0).all()
    assert v.loc[v["fraude"] == 1, "etranger"].mean() > v.loc[v["fraude"] == 0, "etranger"].mean()
