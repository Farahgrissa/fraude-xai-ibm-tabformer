import plotly.graph_objects as go
import streamlit as st
from commun import COULEURS, MODELES, config, configurer, scores, tableau
from sklearn.metrics import average_precision_score, precision_recall_curve

configurer("Modèles", "🤖")
c = config()

st.title("🤖 Module 3 – Modélisation")
st.markdown("""
| Famille | Méthode | Entraînement |
|---|---|---|
| Supervisé | **XGBoost** (retenu) | avec la stratégie de déséquilibre retenue |
| Supervisé | Régression logistique pondérée | variables standardisées, `class_weight="balanced"` |
| Non supervisé | Isolation Forest | toutes les transactions d'entraînement, **sans** étiquette |
| Non supervisé | Autoencodeur | transactions **normales** seulement (détection de nouveauté) ; score = erreur de reconstruction |
""")

cl = tableau("classement.csv", header=[0, 1], index_col=0)
st.subheader("Tableau comparatif")
st.caption(f"k = {c['BUDGET_JOUR']} alertes par jour × nombre de jours de la période. "
           "PR-AUC d'un classement au hasard = taux de fraude.")
if cl is not None:
    st.dataframe(cl.style.format("{:.4f}").highlight_max(axis=0, color="#fadbd8"), width="stretch")

rac = tableau("raccourci_marchand.csv", index_col=0)
if rac is not None:
    with st.expander("Raccourci écarté : la variable « risque du marchand »"):
        st.dataframe(rac, width="stretch")
        st.markdown("Un taux de fraude par marchand (appris sur l'entraînement) fait passer la PR-AUC de 0,31 à 0,47. "
                    "Mais 85 % des fraudes futures ont lieu chez un marchand déjà fraudé : le modèle reconnaîtrait des "
                    "marchands au lieu d'un comportement, et serait aveugle face à un nouveau marchand compromis. "
                    "Cette variable est donc **exclue** du modèle principal.")

st.subheader("Courbes précision-rappel superposées")
partie = st.radio("Période", ["validation", "test"], horizontal=True)
s = scores()
s = s[s["partie"] == partie]
y = s["fraude"].to_numpy()
fig = go.Figure()
for nom in [m for m in MODELES if m in s.columns]:
    p, r, _ = precision_recall_curve(y, s[nom])
    pas = max(1, len(p) // 3000)  # Allège le tracé
    fig.add_scatter(x=r[::pas], y=p[::pas], name=f"{nom} ({average_precision_score(y, s[nom]):.3f})",
                    line=dict(color=COULEURS.get(nom)))
fig.add_hline(y=y.mean(), line_dash="dot", line_color="grey", annotation_text="hasard")
fig.update_layout(xaxis_title="rappel", yaxis_title="précision", height=450, legend=dict(x=0.6, y=0.98))
st.plotly_chart(fig, width="stretch")

st.subheader("Score des fraudes et des transactions normales")
nom = st.selectbox("Modèle", [m for m in MODELES if m in s.columns])
rang = s[nom].rank(pct=True)
fig = go.Figure()
for k, lab, coul in [(0, "normale", "#5d6d7e"), (1, "fraude", "#c0392b")]:
    fig.add_histogram(x=rang[s["fraude"] == k], name=lab, histnorm="probability", nbinsx=50, marker_color=coul,
                      opacity=0.6)
fig.update_layout(barmode="overlay", xaxis_title="rang du score (centile)", height=320, yaxis_type="log")
st.plotly_chart(fig, width="stretch")
st.caption("Un bon détecteur place les fraudes dans les derniers centiles. Les méthodes non supervisées repèrent "
           "l'inhabituel, qui n'est pas toujours frauduleux.")
