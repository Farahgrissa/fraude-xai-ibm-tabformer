import json
import os

import requests
import streamlit as st
from commun import ARTEFACTS, configurer

configurer("API", "🔌")
st.title("🔌 Module 6 – API FastAPI")

defaut = os.environ.get("API_URL", "")
try:
    defaut = st.secrets.get("API_URL", defaut)
except Exception:  # Pas de fichier secrets.toml en local
    pass
url = st.text_input("Adresse de l'API", defaut or "http://localhost:8000").rstrip("/")

st.markdown("""
| Route | Rôle |
|---|---|
| `GET /health` | état du service et seuil |
| `GET /modele` | modèle, stratégie, seuil, coût d'une alerte, performances |
| `POST /score` | transaction brute + historique de la carte → score et décision |
| `POST /explain` | idem + codes motifs, contributions SHAP et variables calculées |
| `POST /score/variables` | notation à partir de variables déjà construites |
""")

c1, c2 = st.columns(2)
if c1.button("Tester /health"):
    try:
        st.json(requests.get(f"{url}/health", timeout=60).json())
    except requests.RequestException as e:
        st.error(f"API injoignable : {e}. Sur Render, le premier appel peut prendre une minute (mise en veille).")
if c2.button("Voir /modele"):
    try:
        st.json(requests.get(f"{url}/modele", timeout=60).json())
    except requests.RequestException as e:
        st.error(f"API injoignable : {e}")

st.subheader("Expliquer une vraie alerte du test")
chemin = ARTEFACTS / "exemple_api.json"
if chemin.exists():
    ex = json.loads(chemin.read_text(encoding="utf-8"))
    d = ex["demande"]
    st.caption(f"Carte {d['carte']} : {len(d['historique'])} transactions d'historique, puis la transaction à noter. "
               f"Score attendu : {ex['score_attendu']:.4f}.")
    st.json(d["transaction"])
    if st.button("Envoyer à POST /explain", type="primary"):
        try:
            rep = requests.post(f"{url}/explain", json=d, timeout=120)
            rep.raise_for_status()
            r = rep.json()
            st.metric("Score", f"{r['score']:.4f}", "ALERTE" if r["alerte"] else "pas d'alerte",
                      delta_color="inverse" if r["alerte"] else "off")
            st.markdown("**Codes motifs**")
            for cm in r["codes_motifs"]:
                st.markdown(f"- {cm['motif']} (+{cm['contribution']:.2f})")
            with st.expander("Réponse complète"):
                st.json(r)
        except requests.RequestException as e:
            st.error(f"Erreur : {e}")

st.subheader("Appel depuis un terminal")
st.code(f"""jq .demande artefacts/exemple_api.json | curl -X POST {url}/explain \\
  -H "Content-Type: application/json" -d @-""", language="bash")
st.caption(f"Documentation interactive : {url}/docs")
