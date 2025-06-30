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
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor

# ─────── Configuration ───────
st.set_page_config(page_title="Data Insight 360", layout="wide")
st.title("Data Insight 360")
st.markdown("Importe un fichier CSV pour démarrer l’analyse et l'entraînement automatique d’un modèle.")

# ─────── Upload ───────
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

# ─────── Suggestion de variable cible ───────
if st.checkbox("Suggérer la variable cible (basée sur la corrélation)"):
    try:
        correlations = df.select_dtypes(include=np.number).corr()
        suggestion = correlations.sum().sort_values(ascending=False).index[1]
        st.info(f"Suggestion : Essayez '{suggestion}' comme variable cible")
    except:
        st.warning("Impossible de suggérer une variable cible")

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

# ─────── Machine Learning ───────
st.subheader("Modélisation automatique")
target_column = st.selectbox("Choisir la variable cible à prédire", df.columns)
model_choice = st.selectbox("Choisir un modèle", ["Random Forest", "XGBoost", "SVM"])

if target_column:
    try:
        X = df.drop(columns=[target_column])
        y = df[target_column]

        X = pd.get_dummies(X)
        df_cleaned = pd.concat([X, y], axis=1).dropna()

        X = df_cleaned.drop(columns=[target_column])
        y = df_cleaned[target_column]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        classification = y.dtype == 'object' or y.nunique() < 20

        if model_choice == "Random Forest":
            model = RandomForestClassifier() if classification else RandomForestRegressor()
        elif model_choice == "XGBoost":
            model = XGBClassifier() if classification else XGBRegressor()
        else:
            model = SVC() if classification else SVR()

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # ─────── Résultats ───────
        st.markdown(f"### Résultats du modèle : {model_choice} ({'Classification' if classification else 'Régression'})")

        if classification:
            acc = accuracy_score(y_test, y_pred)
            st.write(f"Accuracy : {acc:.2f}")
            st.text(classification_report(y_test, y_pred))
            cm = confusion_matrix(y_test, y_pred)
            fig_cm, ax_cm = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
            ax_cm.set_xlabel("Prédiction")
            ax_cm.set_ylabel("Réel")
            st.pyplot(fig_cm)
        else:
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            st.write(f"RMSE : {rmse:.2f}")

        # ─────── Historique ───────
        if "historique_modeles" not in st.session_state:
            st.session_state.historique_modeles = []

        st.session_state.historique_modeles.append({
            "Modèle": model_choice,
            "Type": "Classification" if classification else "Régression",
            "Score": acc if classification else rmse
        })

        st.subheader("Historique des performances")
        st.dataframe(pd.DataFrame(st.session_state.historique_modeles))

        # ─────── Importance ───────
        if hasattr(model, "feature_importances_"):
            importance_df = pd.DataFrame({
                "Variable": X.columns,
                "Importance": model.feature_importances_
            }).sort_values(by="Importance", ascending=False).head(10)

            st.subheader("Variables les plus importantes")
            st.dataframe(importance_df)
            fig_imp, ax_imp = plt.subplots()
            sns.barplot(x="Importance", y="Variable", data=importance_df, ax=ax_imp)
            st.pyplot(fig_imp)

        # ─────── Interprétation automatique ───────
        with st.expander("Interprétation des résultats"):
            if classification:
                if acc > 0.8:
                    st.write("Très bonne précision. Le modèle est fiable.")
                elif acc > 0.6:
                    st.write("Précision moyenne. Envisagez d'ajuster les variables.")
                else:
                    st.write("Faible précision. Essayez un autre modèle ou améliorez les données.")
            else:
                std_dev = df[target_column].std()
                if rmse < std_dev:
                    st.write("Erreur faible par rapport à la variance des données. Bon modèle.")
                else:
                    st.write("Erreur importante. Vérifiez les données ou changez de modèle.")

        # ─────── Téléchargement ───────
        st.subheader("Télécharger les prédictions")
        result_df = X_test.copy()
        result_df["Valeur réelle"] = y_test
        result_df["Prédiction"] = y_pred
        csv = result_df.to_csv(index=False).encode("utf-8")
        st.download_button("Télécharger CSV", csv, "predictions.csv", "text/csv")

        # ─────── Sauvegarde ───────
        if st.button("Sauvegarder le modèle"):
            joblib.dump(model, "modele_entraine.pkl")
            st.success("Modèle sauvegardé sous 'modele_entraine.pkl'")

    except Exception as e:
        st.error(f"Erreur : {e}")
