import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from commun import MODELES, config, configurer, dollars, pourcent, scores, tableau

from fraude_xai.cout import courbe_cout, metriques_seuil, seuil_optimal, seuil_pour_budget

configurer("Seuil et coût", "🎚️")
c = config()

st.title("🎚️ Module 4 – Évaluation et seuil")
st.markdown("**Coût(τ) = montants des fraudes manquées + coût d'analyse × nombre d'alertes.** Le seuil retenu est "
            "celui qui minimise ce coût **en validation** ; il est ensuite appliqué tel quel au test.")

s_all = scores()
with st.sidebar:
    st.header("Réglages")
    partie = st.radio("Période", ["validation", "test"], index=1, horizontal=True)
    nom = st.selectbox("Modèle", [m for m in MODELES if m in s_all.columns])
    cout_alerte = st.slider("Coût d'analyse d'une alerte ($)", 0.5, 50.0, float(c["cout_alerte"]), 0.5)
    mode = st.radio("Fixer", ["le seuil τ", "le budget (alertes par jour)"])

s = s_all[s_all["partie"] == partie]
y, sc, m = s["fraude"].to_numpy(), s[nom].to_numpy(), s["montant"].to_numpy()
jours = max(1, s["date"].dt.normalize().nunique())



@st.cache_data
def tau_validation(nom: str, cout_alerte: float) -> float:
    if nom == "XGBoost" and cout_alerte == c["cout_alerte"]:
        return float(c["seuil"])
    va = s_all[s_all["partie"] == "validation"]
    return seuil_optimal(va["fraude"].to_numpy(), va[nom].to_numpy(), va["montant"].to_numpy(), cout_alerte)


tau_val = tau_validation(nom, cout_alerte)

with st.sidebar:
    if mode == "le seuil τ":
        # Grille logarithmique : les scores utiles s'étalent sur plusieurs ordres de grandeur
        lo, hi = float(np.quantile(sc, 0.5)), float(np.quantile(sc, 0.99995))
        grille = np.unique(np.r_[np.geomspace(max(lo, 1e-7), max(hi, lo * 10), 300), tau_val])
        tau = st.select_slider("Seuil τ (échelle log)", options=[float(g) for g in grille],
                               value=float(grille[np.argmin(np.abs(grille - tau_val))]),
                               format_func=lambda v: f"{v:.5f}")
    else:
        budget = st.slider("Alertes par jour", 1, 200, int(c["BUDGET_JOUR"]))
        tau = seuil_pour_budget(sc, budget * jours)
    st.caption(f"τ optimal en validation pour ce modèle et ce coût : {tau_val:.5f}")

r = metriques_seuil(y, sc, m, tau, cout_alerte)
k = st.columns(6)
k[0].metric("Alertes", f"{r['alertes']:,}".replace(",", " "), f"{r['alertes'] / jours:.1f} par jour", delta_color="off")
k[1].metric("Précision", pourcent(r["précision"]))
k[2].metric("Rappel", pourcent(r["rappel"]))
k[3].metric("F2", f"{r['F2']:.3f}")
k[4].metric("MCC", f"{r['MCC']:.3f}")
k[5].metric("Coût total", dollars(r["coût total ($)"]),
            f"{pourcent(1 - r['coût total ($)'] / r['montant fraudé total ($)'], 0)} évité", delta_color="normal")

c1, c2 = st.columns([3, 2])
cc = courbe_cout(y, sc, m, cout_alerte)
cc = cc[cc["alertes"] <= max(5 * r["alertes"], 20 * jours)]
fig = go.Figure()
fig.add_scatter(x=cc["alertes"], y=cc["coût total ($)"], name="coût total", line=dict(color="#c0392b", width=2))
fig.add_scatter(x=cc["alertes"], y=cc["fraudes manquées ($)"], name="fraudes manquées", line=dict(dash="dot"))
fig.add_scatter(x=cc["alertes"], y=cc["coût d'analyse ($)"], name="coût d'analyse", line=dict(dash="dot"))
fig.add_vline(x=r["alertes"], line_dash="dash", line_color="grey", annotation_text="seuil actuel")
fig.update_layout(xaxis_title=f"nombre d'alertes ({partie})", yaxis_title="$", height=380,
                  title="Coût selon le nombre d'alertes", legend=dict(orientation="h", y=-0.2))
c1.plotly_chart(fig, width="stretch")

mat = pd.DataFrame([[r["VP"], r["FN"]], [r["FP"], r["VN"]]], index=["fraude réelle", "normale réelle"],
                   columns=["alerte", "pas d'alerte"])
c2.markdown("**Matrice de confusion**")
c2.dataframe(mat, width="stretch")
c2.markdown(f"""
| Poste | Montant |
|---|---|
| Fraudes manquées | {dollars(r['fraudes manquées ($)'])} |
| Analyse des alertes | {dollars(r["coût d'analyse ($)"])} |
| **Coût total** | **{dollars(r['coût total ($)'])}** |
| Sans modèle (toutes les fraudes passent) | {dollars(r['montant fraudé total ($)'])} |
""")

st.subheader("Seuil retenu : validation puis test")
au = tableau("au_seuil.csv", index_col=0)
if au is not None:
    cols = ["seuil", "alertes", "alertes / jour", "précision", "rappel", "F2", "MCC", "coût total ($)", "sans modèle ($)"]
    st.dataframe(au[[x for x in cols if x in au.columns]].style
                 .format("{:.4f}", subset=["seuil", "précision", "rappel", "F2", "MCC"])
                 .format("{:,.0f}", subset=["alertes", "coût total ($)", "sans modèle ($)"])
                 .format("{:.1f}", subset=["alertes / jour"]), width="stretch")
    st.caption("« τ oracle » : seuil qui aurait été optimal sur le test, inconnu en pratique. L'écart mesure la perte "
               "due au transfert du seuil d'une année à l'autre.")
st_ = tableau("seuil_tous.csv", index_col=0)
if st_ is not None:
    st.markdown("**Tous les modèles, seuil choisi en validation, résultats sur le test**")
    st.dataframe(st_.style.format("{:.4f}", subset=["précision", "rappel", "F2", "MCC"])
                 .format("{:,.0f}", subset=["alertes", "coût total ($)"]), width="stretch")
sens = tableau("sensibilite_cout.csv")
if sens is not None:
    with st.expander("Sensibilité du seuil au coût d'une alerte (validation)"):
        st.dataframe(sens, hide_index=True, width="stretch")
