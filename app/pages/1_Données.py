import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from commun import FRAUDE, NORMAL, configurer, dollars, eda, nombre, pourcent

configurer("Données", "📊")
e = eda()
r = e["resume"]

st.title("📊 Module 1 – Données")
st.markdown("Échantillon IBM TabFormer : 500 clients tirés au hasard (stratifiés selon la présence d'une fraude) "
            "avec **tout leur historique**, de 2015 à 2019.")

k = st.columns(5)
k[0].metric("Transactions", nombre(r["transactions"]))
k[1].metric("Fraudes", nombre(r["fraudes"]))
k[2].metric("Taux de fraude", pourcent(r["taux"], 3))
k[3].metric("Montant fraudé", dollars(r["montant_fraude"]))
k[4].metric("Cartes", nombre(r["cartes"]))

part = pd.DataFrame(e["partie"]).set_index("partie").loc[["entraînement", "validation", "test"]]
part["période"] = ["2015 – 2017", "2018", "2019"]
part["taux"] = (part["taux"] * 100).round(3).astype(str) + " %"
st.dataframe(part[["période", "transactions", "fraudes", "taux"]], width="stretch")

st.subheader("Évolution du taux de fraude")
mois = pd.DataFrame(e["mois"])
fig = go.Figure()
fig.add_bar(x=mois["mois"], y=mois["transactions"], name="transactions", marker_color="#e5e7e9")
fig.add_scatter(x=mois["mois"], y=mois["taux"] * 100, name="taux de fraude (%)", line=dict(color=FRAUDE, width=2.5),
                mode="lines+markers", yaxis="y2")
for x, t in [("2018-01", "validation"), ("2019-01", "test")]:
    fig.add_vline(x=x, line_dash="dash", line_color="grey")
    fig.add_annotation(x=x, y=1, yref="paper", text=t, showarrow=False, xanchor="left")
fig.update_layout(yaxis=dict(title="transactions", side="right", showgrid=False),
                  yaxis2=dict(title="taux de fraude (%)", overlaying="y", side="left"),
                  height=360, legend=dict(orientation="h", y=1.12), margin=dict(t=30))
st.plotly_chart(fig, width="stretch")
st.caption(f"{(mois['fraudes'] == 0).sum()} mois sur {len(mois)} sans aucune fraude : la fraude arrive par campagnes "
           "courtes et intenses.")

st.subheader("Montants")
c1, c2 = st.columns(2)
bords = np.linspace(-1, 4, 61)
centres = 10 ** ((bords[:-1] + bords[1:]) / 2)
fig = go.Figure()
for k_, nom, coul in [("0", "normale", NORMAL), ("1", "fraude", FRAUDE)]:
    fig.add_scatter(x=centres, y=e["hist_montant"][k_], name=nom, line=dict(color=coul), fill="tozeroy", opacity=0.5)
fig.update_layout(xaxis=dict(type="log", title="montant ($, échelle log)"), yaxis_title="densité", height=330,
                  title="Distribution des montants par classe", margin=dict(t=40))
c1.plotly_chart(fig, width="stretch")
tr = pd.DataFrame(e["tranche_montant"])
ordre = ["≤0", "0-10", "10-50", "50-100", "100-250", "250-500", "500-1000", ">1000"]
tr = tr.set_index("tranche").reindex(ordre).reset_index()
fig = px.bar(tr, x="tranche", y=tr["taux"] * 100, labels={"y": "taux de fraude (%)", "tranche": "montant ($)"},
             title="Taux de fraude par tranche de montant", color_discrete_sequence=[FRAUDE], height=330)
c2.plotly_chart(fig, width="stretch")

st.subheader("Heure et jour")
c1, c2 = st.columns(2)
h = pd.DataFrame(e["heure"])
c1.plotly_chart(px.bar(h, x="heure", y=h["taux"] * 100, labels={"y": "taux de fraude (%)"}, height=300,
                       title="Taux de fraude par heure", color_discrete_sequence=[FRAUDE]), width="stretch")
j = pd.DataFrame(e["jour"])
j["jour"] = j["jour"].map(dict(enumerate(["lun", "mar", "mer", "jeu", "ven", "sam", "dim"])))
c2.plotly_chart(px.bar(j, x="jour", y=j["taux"] * 100, labels={"y": "taux de fraude (%)"}, height=300,
                       title="Taux de fraude par jour", color_discrete_sequence=[FRAUDE]), width="stretch")

st.subheader("Canal, lieu et catégorie de marchand")
c1, c2 = st.columns(2)
for col, cle, titre in [(c1, "canal", "Par canal"), (c2, "lieu", "Par lieu")]:
    t = pd.DataFrame(e[cle]).sort_values("taux")
    fig = px.bar(t, y=cle, x=t["taux"] * 100, orientation="h", text=(t["taux"] * 100).round(3).astype(str) + " %",
                 labels={"x": "taux de fraude (%)", cle: ""}, title=titre, height=260,
                 color_discrete_sequence=[FRAUDE])
    col.plotly_chart(fig, width="stretch")
st.caption("À l'étranger, le taux de fraude est environ 490 fois plus élevé qu'en magasin aux États-Unis.")

m = pd.DataFrame(e["mcc"])
m["taux (%)"] = (m["taux"] * 100).round(2)
c1, c2 = st.columns([3, 2])
c1.plotly_chart(px.bar(m.head(15).sort_values("taux"), y="mcc", x="taux (%)", orientation="h", height=420,
                       title="Catégories de marchand les plus fraudées (≥ 200 transactions)",
                       color_discrete_sequence=[FRAUDE]), width="stretch")
err = pd.DataFrame(e["erreurs"])
err["taux (%)"] = (err["fraudes"] / err["transactions"].clip(lower=1) * 100).round(2)
c2.markdown("**Erreurs à l'autorisation**")
c2.dataframe(err.sort_values("taux (%)", ascending=False), hide_index=True, width="stretch")
