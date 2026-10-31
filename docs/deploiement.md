# Déploiement

Deux services, comme pour le Challenge 1 :

| Service | Hébergeur | Ce qui est déployé |
|---|---|---|
| API FastAPI | Render (Docker, offre gratuite) | `Dockerfile` : package + API + `modele_xgb.json` + `config.json` |
| Application Streamlit | Streamlit Community Cloud | `app/Accueil.py`, lit tout le dossier `artefacts/` |

## 0. Préparer le dépôt

1. Exécuter `notebooks/04_modules_4_5.ipynb` dans Colab. La dernière cellule télécharge `artefacts.zip`.
2. Dézipper son contenu dans le dossier `artefacts/` du dépôt (à côté de `LISEZMOI.md`, `strategies_v1.csv`,
   `raccourci_marchand.csv`).
3. Vérifier en local :
   ```bash
   pytest -q && ruff check .
   streamlit run app/Accueil.py
   uvicorn api.main:app
   ```
4. Pousser sur GitHub :
   ```bash
   git init && git add . && git commit -m "Détection de fraude explicable : application, API, notebooks"
   git branch -M main
   git remote add origin https://github.com/Farahgrissa/fraude-xai-ibm-tabformer.git
   git push -u origin main
   ```
   Le fichier le plus lourd (`scores.parquet`) fait quelques dizaines de Mo : en dessous de la limite de 100 Mo de GitHub.

## 1. API sur Render

1. render.com → **New → Web Service** → connecter le dépôt `fraude-xai-ibm-tabformer`.
2. **Language : Docker**, branche `main`, offre **Free**.
3. Variable d'environnement : aucune obligatoire (`PORT` est fourni par Render).
4. **Health Check Path** : `/health`.
5. Créer. Au bout de quelques minutes : `https://fraude-xai-ibm-tabformer-api.onrender.com/docs`.

L'offre gratuite met le service en veille après 15 minutes sans requête : le premier appel prend environ une minute.

## 2. Application sur Streamlit Community Cloud

1. share.streamlit.io → **Create app** → dépôt `Farahgrissa/fraude-xai-ibm-tabformer`, branche `main`,
   fichier principal `app/Accueil.py`.
2. **Advanced settings** : Python 3.13 ; dans **Secrets** :
   ```toml
   API_URL = "https://fraude-xai-ibm-tabformer-api.onrender.com"
   ```
3. Deploy. Adresse proposée : `fraude-xai-ibm-tabformer.streamlit.app`.

## 3. Finaliser

- Reporter les deux liens dans le tableau en haut du `README.md`.
- Vérifier la page **API** de l'application : bouton « Envoyer à POST /explain ».
- La CI GitHub Actions lance ruff, les tests, puis construit et teste l'image Docker.
