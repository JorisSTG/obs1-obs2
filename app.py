import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# ---- STYLE sombre pour se fondre avec le thème Streamlit ----
plt.style.use("dark_background")
plt.rcParams.update({
    "figure.facecolor": "none",
    "axes.facecolor": "none",
    "savefig.facecolor": "none",
    "axes.edgecolor": "#FFFFFF",
    "axes.labelcolor": "#FFFFFF",
    "xtick.color": "#DDDDDD",
    "ytick.color": "#DDDDDD",
    "text.color": "#FFFFFF",
})

st.title("Comparaison : Modèle 1 / Modèle 2")
st.markdown(
    """
    L’objectif de cette application est de comparer deux jeux de données de température (modèle 1 et modèle 2).
    """,
    unsafe_allow_html=True
)

# -------- Paramètres --------
heures_par_mois = [744, 672, 744, 720, 744, 720, 744, 744, 720, 744, 720, 744]
percentiles_list = [10, 25, 50, 75, 90]
couleur_modele = "goldenrod"
couleur_TRACC = "lightgray"
vmaxT = 5
vminT = -5
vmaxP = 100
vminP = 50
vmaxH = 100
vminH = -100
vmaxDJU = 150
vminDJU = -150

# -------- Noms des mois --------
mois_noms = {
    1: "01 - Janvier", 2: "02 - Février", 3: "03 - Mars",
    4: "04 - Avril", 5: "05 - Mai", 6: "06 - Juin",
    7: "07 - Juillet", 8: "08 - Août", 9: "09 - Septembre",
    10: "10 - Octobre", 11: "11 - Novembre", 12: "12 - Décembre"
}

# -------- Upload des fichiers CSV --------
uploaded_model1 = st.file_uploader("Déposer le fichier CSV du modèle 1 (colonne unique T°C) :", type=["csv"])
uploaded_model2 = st.file_uploader("Déposer le fichier CSV du modèle 2 (colonne unique T°C) :", type=["csv"])

if uploaded_model1 and uploaded_model2:
    st.markdown("")

    # -------- Lecture des fichiers CSV --------
    model1_values = pd.read_csv(uploaded_model1, header=0).iloc[:, 0].values
    model2_values = pd.read_csv(uploaded_model2, header=0).iloc[:, 0].values

    # -------- Vérification de la longueur des données --------
    if len(model1_values) != 8760 or len(model2_values) != 8760:
        st.error("Les fichiers doivent contenir 8760 valeurs (une par heure pour une année).")
    else:
        st.success("Les deux fichiers contiennent bien 8760 valeurs.")

    # -------- RMSE --------
    def rmse(a, b):
        min_len = min(len(a), len(b))
        a_sorted = np.sort(a[:min_len])
        b_sorted = np.sort(b[:min_len])
        return np.sqrt(np.nanmean((a_sorted - b_sorted) ** 2))

    # -------- Précision basée sur les écarts de percentiles --------
    def precision_ecarts_percentiles(a, b):
        if len(a) == 0 or len(b) == 0:
            return np.nan
        percentiles = np.arange(1, 100)
        pa = np.percentile(a, percentiles)
        pb = np.percentile(b, percentiles)

        diff_moyenne = np.mean(np.abs(pa - pb))
        scale = np.std(pb)

        if scale == 0:
            return 100.0

        score = 100 * (1 - diff_moyenne / (2 * scale))
        score = max(0, min(100, score))

        return round(score, 2)

    # -------- Boucle sur les mois --------
    results_rmse = []
    start_idx = 0

    for mois_num, nb_heures in enumerate(heures_par_mois, start=1):
        mois = mois_noms[mois_num]
        mod1_mois = model1_values[start_idx:start_idx + nb_heures]
        mod2_mois = model2_values[start_idx:start_idx + nb_heures]
        val_rmse = rmse(mod1_mois, mod2_mois)
        pct_precision = precision_ecarts_percentiles(mod1_mois, mod2_mois)
        results_rmse.append({
            "Mois": mois,
            "RMSE (°C)": round(val_rmse, 2),
            "Précision percentile (%)": pct_precision
        })
        start_idx += nb_heures

    # -------- DataFrame final --------
    df_rmse = pd.DataFrame(results_rmse)
    df_rmse_styled = (
        df_rmse.style
        .background_gradient(subset=["Précision percentile (%)"], cmap="RdYlGn", vmin=vminP, vmax=vmaxP, axis=None)
        .format({"Précision percentile (%)": "{:.2f}", "RMSE (°C)": "{:.2f}"})
    )

    st.subheader("Précision du modèle 1 par rapport au modèle 2 : RMSE et précision via écarts des percentiles")
    st.dataframe(df_rmse_styled, hide_index=True)
