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
    L’objectif de cette application est d’évaluer la précision de données météorologiques en les comparant à des données de référence.
    """,
    unsafe_allow_html=True
)

# -------- Paramètres --------
scenarios = ["2", "2_VC", "2-7", "2-7_VC", "4", "4_VC"]
villes = ["AGEN", "CARPENTRAS", "MACON", "MARIGNANE", "NANCY", "RENNES", "TOURS", "TRAPPES"]
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

# -------- Choix scénario et ville --------
scenario_sel = st.selectbox("Choisir le scénario :", scenarios)
ville_sel = st.selectbox("Choisir la ville :", villes)

# -------- Upload des fichiers CSV --------
uploaded_model1 = st.file_uploader("Déposer le fichier CSV du modèle 1 (colonne unique T°C) :", type=["csv"])
uploaded_model2 = st.file_uploader("Déposer le fichier CSV du modèle 2 (colonne unique T°C) :", type=["csv"])

if uploaded_model1 and uploaded_model2:
    st.markdown("")

    # -------- Lecture des fichiers CSV --------
    model_values = pd.read_csv(uploaded_model1, header=0).iloc[:, 0].values
    obs_series = pd.read_csv(uploaded_model2, header=0).iloc[:, 0].values

    # -------- Création de df_obs (pour compatibilité avec la suite du code) --------
    df_obs = pd.DataFrame({"T2m": obs_series})
    df_obs["year"] = 2023  # Année fictive pour compatibilité
    df_obs["month_num"] = pd.concat([pd.Series([m] * h) for m, h in enumerate(heures_par_mois, start=1)], ignore_index=True)
    df_obs["month"] = df_obs["month_num"].map(mois_noms)
    df_obs["day"] = pd.concat([pd.Series(range(1, h // 24 + 2)) for h in heures_par_mois], ignore_index=True)[:len(obs_series)]

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
    obs_mois_all = []
    start_idx_model = 0

    for mois_num, nb_heures in enumerate(heures_par_mois, start=1):
        mois = mois_noms[mois_num]
        mod_mois = model_values[start_idx_model:start_idx_model + nb_heures]
        obs_mois_vals = df_obs[df_obs["month_num"] == mois_num]["T2m"].values
        obs_mois_all.append(obs_mois_vals)
        val_rmse = rmse(mod_mois, obs_mois_vals)
        pct_precision = precision_ecarts_percentiles(mod_mois, obs_mois_vals)
        results_rmse.append({
            "Mois": mois,
            "RMSE (°C)": round(val_rmse, 2),
            "Précision percentile (%)": pct_precision
        })
        start_idx_model += nb_heures

    # -------- DataFrame final --------
    df_rmse = pd.DataFrame(results_rmse)
    df_rmse_styled = (
        df_rmse.style
        .background_gradient(subset=["Précision percentile (%)"], cmap="RdYlGn", vmin=vminP, vmax=vmaxP, axis=None)
        .format({"Précision percentile (%)": "{:.2f}", "RMSE (°C)": "{:.2f}"})
    )

    st.subheader("Précision du modèle 1 par rapport au modèle 2 : RMSE et précision via écarts des percentiles")
    st.dataframe(df_rmse_styled, hide_index=True)

    # -------- Suite de votre code --------
    # (Coller ici la suite de votre code original)
    t_sup_thresholds = st.text_input("Seuils Tmax supérieur (°C, séparés par des virgules)", "25,30,35")
    t_inf_thresholds = st.text_input("Seuils Tmin inférieur (°C, séparés par des virgules)", "-5,0,5")
    t_sup_thresholds_list = [int(float(x.strip())) for x in t_sup_thresholds.split(",")]
    t_inf_thresholds_list = [int(float(x.strip())) for x in t_inf_thresholds.split(",")]

    stats_sup = []
    stats_inf = []

    for mois_num, nb_heures in enumerate(heures_par_mois, start=1):
        mois = mois_noms[mois_num]
        idx0 = sum(heures_par_mois[:mois_num-1])
        idx1 = sum(heures_par_mois[:mois_num])
        mod_mois = model_values[idx0:idx1]
        obs_mois = obs_mois_all[mois_num-1]

        # Seuils supérieurs
        for seuil in t_sup_thresholds_list:
            heures_obs = np.sum(obs_mois > seuil)
            nb_heures_mod = np.sum(mod_mois > seuil)
            ecart = nb_heures_mod - heures_obs  # Modèle - TRACC
            stats_sup.append({
                "Mois": mois,
                "Seuil (°C)": f"{seuil}",
                "Heures Modèle": nb_heures_mod,
                "Heures TRACC": heures_obs,
                "Ecart (Modèle - TRACC)": ecart
            })

        # Seuils inférieurs
        for seuil in t_inf_thresholds_list:
            heures_obs = np.sum(obs_mois < seuil)
            nb_heures_mod = np.sum(mod_mois < seuil)
            ecart = nb_heures_mod - heures_obs  # Modèle - TRACC
            stats_inf.append({
                "Mois": mois,
                "Seuil (°C)": f"{seuil}",
                "Heures Modèle": nb_heures_mod,
                "Heures TRACC": heures_obs,
                "Ecart (Modèle - TRACC)": ecart
            })

    # Création des DataFrames
    df_sup = pd.DataFrame(stats_sup)
    df_inf = pd.DataFrame(stats_inf)

    # Conversion en int
    for df in [df_sup, df_inf]:
        df["Heures Modèle"] = df["Heures Modèle"].astype(int)
        df["Heures TRACC"] = df["Heures TRACC"].astype(int)
        df["Ecart (Modèle - TRACC)"] = df["Ecart (Modèle - TRACC)"].astype(int)

    # Style : seuils supérieurs → rouge = plus chaud
    df_sup_styled = (
        df_sup.style
        .background_gradient(subset=["Ecart (Modèle - TRACC)"], cmap="bwr", vmin=vminH, vmax=vmaxH, axis=None)
    )
    st.subheader("Nombre d'heures supérieur au(x) seuil(s)")
    st.dataframe(df_sup_styled, hide_index=True)

    # Style : seuils inférieurs → rouge = plus froid
    df_inf_styled = (
        df_inf.style
        .background_gradient(subset=["Ecart (Modèle - TRACC)"], cmap="bwr_r", vmin=vminH, vmax=vmaxH, axis=None)
    )
    st.subheader("Nombre d'heures inférieur au(x) seuil(s)")
    st.dataframe(df_inf_styled, hide_index=True)

    # -------- Histogrammes par plage de température --------
    st.subheader(f"Histogrammes horaire : Modèle 1 et Modèle 2")
    st.markdown(
        """
        La valeur de chaque barre est égale au total d'heures compris entre [ X°C , X+1°C [
        """,
        unsafe_allow_html=True
    )

    # Bins correspondant à [X, X+1[ pour chaque température entière
    bin_edges = np.arange(-5, 46, 1)
    bin_labels = bin_edges[:-1].astype(int)

    def count_hours_in_bins(temp_hourly, bins):
        counts, _ = np.histogram(temp_hourly, bins=bins)
        return counts

    for mois_num in range(1, 13):
        mois = mois_noms[mois_num]

        # Observations
        obs_hourly = obs_mois_all[mois_num-1]
        obs_counts = count_hours_in_bins(obs_hourly, bin_edges)

        # Modèle
        idx0 = sum(heures_par_mois[:mois_num-1])
        idx1 = sum(heures_par_mois[:mois_num])
        mod_hourly = model_values[idx0:idx1]
        mod_counts = count_hours_in_bins(mod_hourly, bin_edges)

        # Préparer le DataFrame pour le plot
        df_plot = pd.DataFrame({
            "Temp_Num": bin_labels,
            "Température": bin_labels.astype(str),
            "TRACC": obs_counts,
            "Modèle": mod_counts
        }).sort_values("Temp_Num")

        # Création du plot
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.bar(df_plot["Temp_Num"] - 0.25, df_plot["TRACC"], width=0.5, label="Modèle 2", color=couleur_TRACC)
        ax.bar(df_plot["Temp_Num"] + 0.25, df_plot["Modèle"], width=0.5, label="Modèle 1", color=couleur_modele)
        ax.set_title(f"{mois} - Durée en heure par seuil de température")
        ax.set_xlabel("Température (°C)")
        ax.set_ylabel("Durée en heure")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

    # -------- Histogramme annuel par plage de température --------
    st.subheader(f"Histogramme annuel : Modèle 1 et Modèle 2")
    st.markdown(
        """
        La valeur de chaque barre est égale au total d'heures compris entre [ X°C , X+1°C [
        sur l'année entière.
        """,
        unsafe_allow_html=True
    )

    # Bins correspondant à [X, X+1[
    bin_edges = np.arange(-5, 46, 1)
    bin_labels = bin_edges[:-1].astype(int)

    # -------- Regroupement ANNUEL --------
    # Observations : concaténer tous les mois
    obs_hourly_annual = np.concatenate(obs_mois_all)

    # Modèle : toutes les valeurs de l'année
    mod_hourly_annual = model_values

    # Comptages annuels
    obs_counts_annual = count_hours_in_bins(obs_hourly_annual, bin_edges)
    mod_counts_annual = count_hours_in_bins(mod_hourly_annual, bin_edges)
    diff_counts_annual_TRACC = np.maximum(0, obs_counts_annual - mod_counts_annual)
    diff_counts_annual_modele = np.maximum(0, mod_counts_annual - obs_counts_annual)

    # Préparer DataFrame pour le plot
    df_plot_year = pd.DataFrame({
        "Temp_Num": bin_labels,
        "Température": bin_labels.astype(str),
        "TRACC": obs_counts_annual,
        "Modèle": mod_counts_annual
    }).sort_values("Temp_Num")

    # Plot
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.bar(df_plot_year["Temp_Num"] - 0.25, df_plot_year["TRACC"], width=0.5,
           label="Modèle 2", color=couleur_TRACC)
    ax.bar(df_plot_year["Temp_Num"] + 0.25, df_plot_year["Modèle"], width=0.5,
           label="Modèle 1", color=couleur_modele)
    ax.set_title("Année entière - Durée en heures par seuil de température")
    ax.set_xlabel("Température (°C)")
    ax.set_ylabel("Durée en heure")
    ax.legend()

    st.pyplot(fig)
    plt.close(fig)

    # Préparer DataFrame pour le plot
    df_plot_year = pd.DataFrame({
        "Temp_Num": bin_labels,
        "Température": bin_labels.astype(str),
        "Différence absolue modele": diff_counts_annual_modele,
        "Différence absolue TRACC": diff_counts_annual_TRACC
    }).sort_values("Temp_Num")

    # Plot
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.bar(df_plot_year["Temp_Num"], df_plot_year["Différence absolue modele"], width=0.8,
           label="Différence : Modèle 1 > Modèle 2", color=couleur_modele)

    ax.bar(df_plot_year["Temp_Num"], -df_plot_year["Différence absolue TRACC"], width=0.8,
           label="Différence : Modèle 1 < Modèle 2", color=couleur_TRACC)

    ax.set_title("Année entière - Différence en heures par seuil de température")
    ax.set_xlabel("Température (°C)")
    ax.set_ylabel("Durée en heure")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

