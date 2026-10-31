# Artefacts

Produits par `notebooks/04_modules_4_5.ipynb` (dossier `artefacts/` du Drive) :

| Fichier | Contenu | Utilisé par |
|---|---|---|
| `modele_xgb.json` | modèle XGBoost retenu (format JSON portable) | API, application |
| `config.json` | variables, encodages, stratégie, seuil, coût d'une alerte, métriques | API, application |
| `scores.parquet` | scores des 4 modèles sur la validation (2018) et le test (2019) | pages Modèles et Seuil |
| `transactions_app.parquet` | alertes les plus fortes, toutes les fraudes, échantillon aléatoire (avec variables) | page Explications |
| `eda.json` | agrégats de l'exploration | page Données |
| `classement.csv`, `au_seuil.csv`, `seuil_tous.csv`, `sensibilite_*.csv` | tableaux de résultats | pages Modèles et Seuil |
| `exemple_api.json` | une vraie alerte du test avec l'historique brut de sa carte | page API, README |

| `strategies_v1.csv`, `raccourci_marchand.csv` | résultats du Module 3 (stratégies de déséquilibre, raccourci marchand) | pages Variables et Modèles |
Recopiés à la main depuis les sorties du notebook 03 (section B.1 et B.8) : `strategies_v1.csv`, `raccourci_marchand.csv`.
