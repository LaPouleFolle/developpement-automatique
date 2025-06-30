# ─────── Imports ───────
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_squared_error, accuracy_score,
    classification_report, confusion_matrix
)

# ─────── Configuration du thème bleu───────
st.set_page_config(page_title="Information sur les données Arsène MBABEH MEYE", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background-color: #0f1c2e;  /* Bleu nuit */
        color: #e0f7fa;             /* Cyan clair */
    }
    .css-1d391kg { color: #00acc1; } /* Titres accent cyan */
    </style>
    """,
    unsafe_allow_html=True
)



# ─────── Titre ───────
st.title("Information sur les données Arsène MBABEH MEYE")
st.markdown("Importe un fichier CSV pour démarrer l’analyse et l'entraînement automatique d’un modèle.")

# ─────── Upload et chargement ───────
uploaded_file = st.file_uploader("Choisir un fichier CSV", type=["csv"])

if uploaded_file is not None:
    sep = st.selectbox("Choisir le séparateur du fichier", [",", ";", "\t"], index=1)
    try:
        df = pd.read_csv(uploaded_file, sep=sep)
        st.success("Fichier chargé avec succès")
        st.subheader("Aperçu des données")
        st.dataframe(df.head())
    except Exception as e:
        st.error(f"Erreur de chargement : {e}")
        st.stop()
else:
    st.stop()

# ─────── Analyse exploratoire ───────
st.subheader("Visualisations interactives")

plot_type = st.selectbox("Type de graphique", ["Bar", "Box", "Scatter"])
x_col = st.selectbox("Colonne X", df.columns)
y_col = st.selectbox("Colonne Y", df.columns)

try:
    if plot_type == "Bar":
        fig = px.bar(df, x=x_col, y=y_col)
    elif plot_type == "Box":
        fig = px.box(df, x=x_col, y=y_col)
    else:
        fig = px.scatter(df, x=x_col, y=y_col)
    st.plotly_chart(fig)
except Exception as e:
    st.warning(f"Graphique non affiché : {e}")

if st.checkbox("Afficher la heatmap des corrélations"):
    corr = df.select_dtypes(include=np.number).corr()
    fig_corr, ax_corr = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax_corr)
    st.pyplot(fig_corr)

# ─────── Machine Learning automatique ───────
st.subheader("Modélisation automatique")

target_column = st.selectbox("Choisir la variable cible à prédire", df.columns)

if target_column:
    try:
        X = df.drop(columns=[target_column])
        y = df[target_column]

        X = pd.get_dummies(X)
        df_cleaned = pd.concat([X, y], axis=1).dropna()

        X = df_cleaned.drop(columns=[target_column])
        y = df_cleaned[target_column]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if y.dtype == 'object' or y.nunique() < 20:
            model_type = "Classification"
            model = RandomForestClassifier()
        else:
            model_type = "Régression"
            model = RandomForestRegressor()

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # ─────── Résultats du modèle ───────
        st.markdown(f"### Résultats du modèle : {model_type}")

        if model_type == "Classification":
            acc = accuracy_score(y_test, y_pred)
            st.success(f"Accuracy : {acc:.2f}")

            st.markdown("**Rapport de classification :**")
            st.text(classification_report(y_test, y_pred))

            st.markdown("**Matrice de confusion :**")
            cm = confusion_matrix(y_test, y_pred)
            fig_cm, ax_cm = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
            ax_cm.set_xlabel("Prédiction")
            ax_cm.set_ylabel("Réel")
            st.pyplot(fig_cm)

        else:
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            st.success(f"RMSE : {rmse:.2f}")

        # ─────── Importance des variables ───────
        st.markdown("### Variables les plus importantes")
        importance_df = pd.DataFrame({
            "Variable": X.columns,
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=False).head(10)

        st.dataframe(importance_df)

        fig_imp, ax_imp = plt.subplots()
        sns.barplot(x="Importance", y="Variable", data=importance_df, ax=ax_imp)
        st.pyplot(fig_imp)

        # ─────── Export des prédictions ───────
        st.subheader("Télécharger les prédictions")
        result_df = X_test.copy()
        result_df["Valeur réelle"] = y_test
        result_df["Prédiction"] = y_pred

        csv = result_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Télécharger en CSV",
            data=csv,
            file_name="predictions.csv",
            mime="text/csv"
        )

        # ─────── Sauvegarde du modèle ───────
        if st.button("Sauvegarder le modèle"):
            joblib.dump(model, "modele_entraine.pkl")
            st.success("Modèle sauvegardé sous 'modele_entraine.pkl'")

    except Exception as e:
        st.error(f"Une erreur est survenue : {e}")
