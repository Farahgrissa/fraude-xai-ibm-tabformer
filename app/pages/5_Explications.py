import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import shap
import streamlit as st
from commun import config, configurer, dollars, modele, transactions

from fraude_xai.explications import LIBELLES, contributions, phrase
from fraude_xai.variables import CATEGORIELLE, matrice

configurer("Explications", "🔍")
c = config()
mod = modele()
tx = transactions()

st.title("🔍 Module 5 – Explainable AI")
st.caption("Contributions SHAP exactes (TreeSHAP calculé par XGBoost, en log-odds). Les deux variables d'heure "
           "sont regroupées en « heure ».")


@st.cache_data
def contrib_echantillon(partie: str):
    ech = tx[(tx["partie"] == partie) & (tx["groupe"] == "échantillon aléatoire")]
    C, base = contributions(mod.booster, matrice(ech, c))
    return ech, C, base


def valeurs_affichage(df: pd.DataFrame, colonnes) -> pd.DataFrame:
    v = pd.DataFrame(index=df.index)
    for col in colonnes:
        if col == "heure":
            v[col] = (np.arctan2(df["heure_sin"], df["heure_cos"]) / (2 * np.pi) * 24) % 24
        elif col in (CATEGORIELLE, "mcc_risque"):
            v[col] = df[CATEGORIELLE].astype(str).map(c["mcc_risque"]).fillna(c["taux_global"])
        else:
            v[col] = df[col]
    return v.astype(float)


onglet_g, onglet_l = st.tabs(["Explication globale", "Explication d'une alerte"])

with onglet_g:
    partie = st.radio("Période", ["test", "validation"], horizontal=True, key="pg")
    ech, C, base = contrib_echantillon(partie)
    imp = C.abs().mean().sort_values(ascending=True).tail(15)
    fig = px.bar(x=imp.values, y=[LIBELLES.get(v, v) for v in imp.index], orientation="h", height=480,
                 labels={"x": "|SHAP| moyen (log-odds)", "y": ""}, title="Variables les plus influentes",
                 color_discrete_sequence=["#c0392b"])
    st.plotly_chart(fig, width="stretch")
    st.caption("Le paiement en ligne, la catégorie de marchand et la nouveauté du marchand dominent. Le jour de la "
               "semaine et le nombre de transactions sur 30 jours pèsent plus qu'attendu : dans ce jeu synthétique, "
               "les campagnes de fraude se concentrent sur certaines périodes, un effet de calendrier à surveiller.")
    with st.spinner("Beeswarm SHAP…"):
        V = valeurs_affichage(ech, C.columns)
        expl = shap.Explanation(C.to_numpy(), base_values=base, data=V.to_numpy(),
                                feature_names=[LIBELLES.get(x, x) for x in C.columns])
        plt.figure()
        shap.plots.beeswarm(expl, max_display=15, show=False)
        st.pyplot(plt.gcf(), clear_figure=True)
    st.caption(f"{len(ech):,} transactions tirées au hasard. Couleur : valeur de la variable (pour la catégorie de "
               "marchand : son taux de fraude en entraînement).".replace(",", " "))

with onglet_l:
    c1, c2, c3 = st.columns(3)
    partie = c1.radio("Période", ["test", "validation"], horizontal=True, key="pl")
    vue = c2.selectbox("Transactions", ["alertes (score ≥ seuil)", "fraudes manquées", "fausses alertes",
                                        "toutes les fraudes"])
    tri = c3.selectbox("Trier par", ["score décroissant", "montant décroissant", "date"])
    d = tx[tx["partie"] == partie].drop_duplicates(subset=["date", "carte", "montant"])
    al = d["score"] >= c["seuil"]
    d = {"alertes (score ≥ seuil)": d[al], "fraudes manquées": d[(d["fraude"] == 1) & ~al],
         "fausses alertes": d[(d["fraude"] == 0) & al], "toutes les fraudes": d[d["fraude"] == 1]}[vue]
    d = d.sort_values({"score décroissant": "score", "montant décroissant": "montant", "date": "date"}[tri],
                      ascending=tri == "date")
    if d.empty:
        st.info("Aucune transaction dans cette vue.")
        st.stop()
    liste = d.head(300).reset_index(drop=True)
    st.caption(f"{len(d):,} transactions (300 premières affichées).".replace(",", " "))
    lib = liste.apply(lambda r: f"{r['date']:%Y-%m-%d %H:%M} · {r['montant']:.2f} $ · score {r['score']:.3f} · "
                                f"{'FRAUDE' if r['fraude'] else 'normale'} · carte {r['carte']}", axis=1)
    i = st.selectbox("Transaction", range(len(liste)), format_func=lambda k: lib[k])
    ligne = liste.iloc[[i]]
    res = mod.expliquer(ligne)[0]
    r = ligne.iloc[0]

    k = st.columns(4)
    k[0].metric("Score de fraude", f"{res['score']:.3f}", f"seuil {res['seuil']:.4f}", delta_color="off")
    k[1].metric("Décision", "🚨 ALERTE" if res["alerte"] else "✅ pas d'alerte")
    k[2].metric("Étiquette réelle", "fraude" if r["fraude"] else "normale")
    k[3].metric("Montant", dollars(r["montant"]) if abs(r["montant"]) >= 1 else f"{r['montant']:.2f} $")

    st.markdown("#### Codes motifs")
    if res["codes_motifs"]:
        for cm in res["codes_motifs"]:
            st.markdown(f"- **{cm['motif']}** &nbsp; <span style='color:grey'>(+{cm['contribution']:.2f})</span>",
                        unsafe_allow_html=True)
    else:
        st.write("Aucune variable ne pousse nettement vers la fraude.")

    Cl, bl = contributions(mod.booster, matrice(ligne, c))
    Vl = valeurs_affichage(ligne, Cl.columns)
    plt.figure()
    shap.plots.waterfall(shap.Explanation(Cl.iloc[0].to_numpy(), base_values=bl[0], data=Vl.iloc[0].to_numpy(),
                                          feature_names=[LIBELLES.get(x, x) for x in Cl.columns]),
                         max_display=10, show=False)
    st.pyplot(plt.gcf(), clear_figure=True)
    st.caption("Waterfall : de la valeur de base (log-odds moyen du modèle) au score de cette transaction. "
               "Rouge = pousse vers la fraude, bleu = vers le normal.")

    with st.expander("Toutes les variables de la transaction"):
        detail = pd.DataFrame({"variable": [LIBELLES.get(v, v) for v in Cl.columns],
                               "valeur": [phrase(v, r) for v in Cl.columns],
                               "contribution": Cl.iloc[0].round(3).to_numpy()})
        st.dataframe(detail.sort_values("contribution", key=abs, ascending=False), hide_index=True,
                     width="stretch")
