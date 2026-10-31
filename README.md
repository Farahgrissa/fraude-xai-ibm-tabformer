# Détection de fraude explicable – IBM TabFormer

[![CI](https://github.com/Farahgrissa/fraude-xai-ibm-tabformer/actions/workflows/ci.yml/badge.svg)](https://github.com/Farahgrissa/fraude-xai-ibm-tabformer/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.13-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.66-red)
![FastAPI](https://img.shields.io/badge/FastAPI-0.162-teal)

Outil d'aide à la décision pour une équipe anti-fraude qui ne peut examiner qu'un nombre limité de dossiers par jour.
Il **classe les transactions par risque**, **déclenche une alerte** selon un seuil fixé par un **coût en dollars**,
et **explique chaque alerte** par des codes motifs (SHAP).

| Ressource | Lien |
|---|---|
| Application (Streamlit Community Cloud) | [fraude-xai-ibm-tabformer.streamlit.app](https://fraude-xai-ibm-tabformer.streamlit.app) |
| API (FastAPI, Render) | [fraude-xai-ibm-tabformer-api.onrender.com/docs](https://fraude-xai-ibm-tabformer-api.onrender.com/docs) — documentation interactive |
| Notebooks | [`notebooks/`](notebooks/) |

---

## Données

**IBM TabFormer – Credit Card Transactions** (synthétique, Kaggle `ealtman2019/credit-card-transactions`) :
24,4 M de transactions, 2 000 clients, 1991 → 2020.

- **Période retenue** : 2015 → 2019 (2020 n'a aucune fraude ; avant 2015, le paiement par puce n'existe pas encore).
  Décembre 2014 sert de période de chauffe pour l'historique des variables.
- **Échantillon** : 500 clients tirés au hasard, stratifiés selon la présence d'une fraude, avec **tout leur historique**
  (2,6 M de transactions, 3 862 fraudes, 0,147 %).
- **Séparation chronologique** par années entières :

| Partie | Période | Transactions | Fraudes | Taux |
|---|---|---|---|---|
| Entraînement | 2015 – 2017 | 1 571 126 | 2 288 | 0,146 % |
| Validation | 2018 | 526 783 | 705 | 0,134 % |
| Test | 2019 | 527 216 | 869 | 0,165 % |

## Démarche

1. **Exploration** : la fraude arrive par campagnes courtes (14 mois sur 60 sans fraude, 2,8 h médianes entre deux
   fraudes d'une carte). À l'étranger, le taux de fraude est environ 490 fois plus élevé qu'en magasin aux États-Unis.
2. **Variables (33)**, toutes calculées sur le **seul passé** de la carte (`closed="left"`, contrôle anti-fuite) :
   heure cyclique, canal, étranger, erreurs à l'autorisation, catégorie de marchand, vélocité (1 h, 24 h, 30 jours),
   vélocité en contexte (nouveaux marchands, montant sur 24 h), écart au profil (montant / moyenne 30 jours), nouveauté
   (marchand, catégorie, ville jamais vus par la carte).
3. **Déséquilibre** : comparaison en validation de six stratégies ; **aucun rééchantillonnage** donne la meilleure PR-AUC,
   le déséquilibre est géré par le seuil. SMOTE dégrade les résultats.
4. **Modèles** : XGBoost (retenu), régression logistique pondérée, Isolation Forest, autoencodeur.
5. **Seuil** choisi sur la validation en minimisant *montants des fraudes manquées + 5 $ × nombre d'alertes*, puis
   confirmé une seule fois sur le test.
6. **Explications** : SHAP exact (TreeSHAP calculé par XGBoost), waterfall et codes motifs pour chaque alerte.

### Raccourci écarté

85 % des fraudes de validation et de test ont lieu chez un marchand déjà fraudé pendant l'entraînement. Une variable
« risque du marchand » ferait passer la PR-AUC de 0,31 à 0,47, mais le modèle reconnaîtrait des marchands au lieu d'un
comportement. Elle est exclue du modèle principal.

## Résultats

### Classement (budget de 10 alertes par jour, k = 3 650)

| Modèle | PR-AUC val. | PR-AUC test | ROC-AUC test | Précision@k test | Rappel@k test |
|---|---|---|---|---|---|
| **XGBoost** | **0,308** | **0,285** | **0,989** | **0,162** | **0,680** |
| Régression logistique pondérée | 0,091 | 0,119 | 0,983 | 0,120 | 0,503 |
| Autoencodeur | 0,023 | 0,027 | 0,966 | 0,033 | 0,140 |
| Isolation Forest | 0,014 | 0,018 | 0,898 | 0,033 | 0,137 |
| *Hasard* | *0,0013* | *0,0016* | *0,5* | | |

### Seuil choisi par le coût (5 $ par alerte analysée)

| | Seuil τ | Alertes / jour | Précision | Rappel | F2 | MCC | Coût total | Sans modèle |
|---|---|---|---|---|---|---|---|---|
| Validation 2018 (τ choisi) | 0,0037 | 8,4 | 14,5 % | 63,3 % | 0,379 | 0,302 | 33 262 $ | 57 944 $ |
| **Test 2019 (même τ)** | 0,0037 | 8,6 | **18,1 %** | **65,1 %** | **0,429** | **0,342** | **35 332 $** | 74 870 $ |
| Test 2019 (τ oracle) | 0,0039 | 8,3 | 18,5 % | 64,8 % | 0,432 | 0,345 | 34 948 $ | 74 870 $ |

- Sur 2019, le modèle **réduit de 53 % le coût de la fraude** par rapport à l'absence de modèle.
- Le seuil fixé en 2018 se transfère presque parfaitement : il coûte seulement 1 % de plus que le seuil optimal de 2019.
- Le seuil optimal donne environ 8 alertes par jour, cohérent avec le budget de 10 dossiers.
- Les méthodes non supervisées ne réduisent presque pas le coût : l'inhabituel n'est pas toujours frauduleux.
- Codes motifs les plus fréquents en tête des alertes : paiement en ligne (47 %), transaction à l'étranger (24 %),
  nouveau marchand (13 %), catégorie de marchand (13 %).

## Architecture

```
fraude_xai/          package partagé par l'application et l'API
  variables.py       construction des variables sans fuite + encodage
  cout.py            précision@k, coût, seuil optimal, F2, MCC
  explications.py    contributions SHAP, codes motifs en français
  modele.py          chargement du modèle, notation d'une transaction brute
api/main.py          API FastAPI
app/                 application Streamlit (6 pages)
artefacts/           modèle, configuration, scores et agrégats (produits par le notebook 04)
notebooks/           01 découverte · 02 modules 1-2 · 03 variables v2 + module 3 · 04 modules 4-5
tests/               tests sur des données simulées au format TabFormer
```

## Installation et exécution

```bash
python -m venv .venv
# Windows : .venv\Scripts\activate    |    Linux / macOS : source .venv/bin/activate
pip install -r requirements-dev.txt

streamlit run app/Accueil.py                 # application  -> http://localhost:8501
uvicorn api.main:app --reload                # API          -> http://localhost:8000/docs
pytest -q && ruff check .                    # tests et qualité
```

Les notebooks s'exécutent sur Google Colab ; ils lisent et écrivent dans `MyDrive/fraude_tabformer`.

## API

| Route | Rôle |
|---|---|
| `GET /health` | état du service et seuil |
| `GET /modele` | modèle, stratégie, seuil, coût d'une alerte, performances |
| `POST /score` | transaction brute + historique de la carte → score et décision |
| `POST /explain` | idem + codes motifs, contributions SHAP et variables calculées |
| `POST /score/variables` | notation à partir de variables déjà construites |

L'API reconstruit les variables avec **le même code** que l'entraînement. L'historique envoyé doit être l'historique
complet de la carte, car les variables de nouveauté et d'habitude en dépendent.

```bash
jq .demande artefacts/exemple_api.json | curl -X POST http://localhost:8000/explain \
  -H "Content-Type: application/json" -d @-
```

Réponse (extrait) :

```json
{
  "score": 0.93, "alerte": true, "seuil": 0.0xx,
  "codes_motifs": [
    {"variable": "nouveau_marchand", "contribution": 3.35, "motif": "Premier achat de la carte chez ce marchand"},
    {"variable": "etranger", "contribution": 1.2, "motif": "Transaction à l'étranger"}
  ]
}
```

## Déploiement

Voir [`docs/deploiement.md`](docs/deploiement.md) : API sur Render (Docker), application sur Streamlit Community Cloud.

## Limites et perspectives

- Données synthétiques : les campagnes de fraude du simulateur sont plus régulières que dans la réalité.
- Probabilités mal calibrées (sous-estimation d'un facteur 3 à 4) : le seuil est choisi empiriquement ; une
  calibration isotone sur la validation permettrait une décision selon le montant (p × montant > coût d'une alerte).
- Échantillon de 500 clients : entraîner sur les 2 000 clients améliorerait la stabilité.
- Agréger les alertes par carte : une campagne génère plusieurs transactions frauduleuses en quelques heures.

## Références

- Altman, E. *Synthesizing Credit Card Transactions*, ICAIF 2021 ; Padhi et al., *Tabular Transformers for Modeling
  Multivariate Time Series*, ICASSP 2021 (jeu IBM TabFormer).
- Le Borgne et al., *Reproducible Machine Learning for Credit Card Fraud Detection – Practical Handbook*, ULB.
- Lundberg & Lee, *A Unified Approach to Interpreting Model Predictions*, NeurIPS 2017 (SHAP).
