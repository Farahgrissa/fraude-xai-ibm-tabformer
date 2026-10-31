import streamlit as st
from commun import config, configurer, dollars, eda, nombre, pourcent

configurer("Accueil")
c, e = config(), eda()
rv, rt = c["metriques_validation"], c["metriques_test"]

st.title("💳 Détection de fraude explicable")
st.markdown(
    "Outil d'aide à la décision pour une équipe anti-fraude qui ne peut examiner qu'un nombre limité de dossiers "
    "par jour : il **classe les transactions par risque**, **signale les plus suspectes** selon un seuil fixé par "
    "un **coût en dollars**, et **explique chaque alerte** par des codes motifs."
)
st.caption("Données : IBM TabFormer (transactions de cartes bancaires synthétiques), échantillon de 500 clients, "
           "2015 → 2019. Projet Challenge 2 – Data Mining, ENSI 2026-2027.")

st.subheader("Résultats sur le test (2019, jamais vu pendant la mise au point)")
k = st.columns(5)
k[0].metric("Coût de la fraude évité", pourcent(1 - rt["coût total ($)"] / rt["sans modèle ($)"], 0),
            help="Réduction du coût total (fraudes manquées + analyse) par rapport à l'absence de modèle.")
k[1].metric("Rappel", pourcent(rt["rappel"], 0), help="Part des fraudes signalées.")
k[2].metric("Précision", pourcent(rt["précision"], 0), help="Part des alertes qui sont de vraies fraudes.")
k[3].metric("Alertes par jour", f"{rt['alertes / jour']:.1f}")
k[4].metric("Coût total", dollars(rt["coût total ($)"]), f"contre {dollars(rt['sans modèle ($)'])} sans modèle",
            delta_color="off")

st.subheader("Démarche")
st.markdown(f"""
| Étape | Choix |
|---|---|
| Données | {nombre(e['resume']['transactions'])} transactions, {nombre(e['resume']['fraudes'])} fraudes ({pourcent(e['resume']['taux'], 3)}) |
| Séparation | chronologique par années entières : 2015-2017 / 2018 / 2019 |
| Variables | {len(c['VARIABLES'])} variables, dont la vélocité et la nouveauté par carte, calculées sur le seul passé |
| Déséquilibre | {c['STRATEGIE']['nom']} (choisi en validation) |
| Modèle retenu | XGBoost, comparé à une régression logistique, Isolation Forest et un autoencodeur |
| Seuil | τ = {c['seuil']:.4f}, qui minimise le coût en validation avec {c['cout_alerte']:.0f} $ par alerte analysée |
| Explications | SHAP exact (TreeSHAP) et codes motifs pour chaque alerte |
""")

st.subheader("Pages")
st.markdown("""
1. **Données** : taux de fraude, montants, évolution dans le temps, heure, canal, lieu, catégorie.
2. **Variables** : construction sans fuite, séparation chronologique, stratégie contre le déséquilibre.
3. **Modèles** : tableau comparatif et courbes précision-rappel.
4. **Seuil et coût** : curseurs de seuil, de budget et de coût, avec mise à jour en direct.
5. **Explications** : SHAP global, waterfall et codes motifs de chaque alerte.
6. **API** : tester l'API FastAPI déployée.
""")
