import pandas as pd
import streamlit as st
from commun import config, configurer, tableau

configurer("Variables", "🧱")
c = config()

st.title("🧱 Module 2 – Préparation et variables")

st.subheader("Séparation chronologique")
st.markdown("""
Les années entières respectent presque exactement 60 / 20 / 20. Décembre 2014 sert de **période de chauffe** :
il fournit l'historique des premières transactions de 2015, puis il est retiré.

| Partie | Période | Rôle |
|---|---|---|
| Entraînement | 2015 – 2017 | apprentissage, rééchantillonnage, encodages |
| Validation | 2018 | choix de la stratégie, des hyperparamètres, du modèle et du seuil |
| Test | 2019 | confirmation une seule fois, rien n'y est réglé |
""")

st.subheader("Variables")
st.info("Règle : **chaque variable d'une transaction n'utilise que le passé de sa carte**. Les fenêtres glissantes "
        "excluent la transaction en cours (`closed=\"left\"`). Un contrôle recalcule les variables d'une carte en "
        "retirant son futur : le passé ne bouge pas.")
familles = pd.DataFrame([
    ("Temporelles", "heure_sin, heure_cos", "heure en codage cyclique (23 h et 0 h sont proches)"),
    ("Temporelles", "jour_semaine, week_end, nuit", "jour, week-end, transaction entre 0 h et 6 h"),
    ("Transaction", "montant, log_montant, remboursement", "montant brut, logarithme signé, montant négatif"),
    ("Transaction", "en_ligne, bande, etranger", "canal (puce = référence) et transaction à l'étranger"),
    ("Transaction", "err_cvv … err_code_postal, une_erreur", "erreurs à l'autorisation, regroupées en indicateurs"),
    ("Transaction", "mcc_groupe", "catégorie de marchand ; codes de voyage regroupés (aérien, hôtel, location)"),
    ("Vélocité", "nb_tx_1h, nb_tx_24h, nb_tx_30j", "nombre de transactions de la carte sur la fenêtre"),
    ("Vélocité", "nb_err_24h, nb_etranger_24h, nb_nouveaux_marchands_24h, montant_24h",
     "vélocité en contexte (v2) : erreurs, étranger, nouveaux marchands, montant dépensé"),
    ("Écart au profil", "moy_30j, ratio_montant_30j", "montant rapporté à la moyenne des 30 derniers jours"),
    ("Écart au profil", "h_depuis_prec", "temps écoulé depuis la transaction précédente (heures)"),
    ("Nouveauté", "nouveau_marchand, nouvelle_categorie, nouvelle_ville",
     "premier achat de la carte chez ce marchand, dans cette catégorie, dans cette ville (v2)"),
    ("Habitude", "part_en_ligne_passee", "part des achats en ligne dans l'historique de la carte"),
], columns=["famille", "variables", "définition"])
st.dataframe(familles, hide_index=True, width="stretch")

st.subheader("Corrections apportées après l'exploration (variables v2)")
st.markdown("""
- `ville_differente_prec` donnait un signal **inversé** : un fraudeur enchaîne les achats dans la *même* ville étrangère.
  Remplacée par `nouvelle_ville`.
- Le nombre de transactions sur 1 h avait un effet non monotone : ajout de vélocités **en contexte**.
- 85 % des fraudes de validation et de test ont lieu chez un marchand déjà fraudé en entraînement :
  l'**identité du marchand est exclue** (raccourci), seule la nouveauté pour la carte est gardée.
""")

st.subheader("Stratégie contre le déséquilibre")
st.markdown(f"Appliquée **uniquement à l'entraînement**, comparée sur la PR-AUC de validation avec XGBoost. "
            f"Retenue : **{c['STRATEGIE']['nom']}**.")
strat = tableau("strategies_v1.csv")
if strat is not None:
    st.dataframe(strat, hide_index=True, width="stretch")
    st.caption("Sans rééchantillonnage, XGBoost garde la meilleure PR-AUC : le déséquilibre est géré par le seuil, "
               "choisi par le coût. SMOTE dégrade nettement les résultats, comme l'annonce le cours pour les arbres "
               "boostés.")
else:
    st.caption("Tableau détaillé des stratégies disponible dans le notebook 03 (section B.1).")
if c.get("BETA", 1) != 1:
    st.markdown(f"Les probabilités sont **corrigées** après rééchantillonnage (β = {c['BETA']:.5f}) : "
                "p = β·pₛ / (β·pₛ − pₛ + 1), pour que le calcul du coût reste juste.")
