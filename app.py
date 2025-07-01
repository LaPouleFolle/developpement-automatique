# ─────── Imports des packages ───────
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.metrics import (
    mean_squared_error, accuracy_score,
    classification_report, confusion_matrix
)

try:
    from xgboost import XGBClassifier, XGBRegressor
    xgb_installed = True
except ImportError:
    xgb_installed = False

# ─────── config du titre ───────
st.set_page_config(page_title="Développé par Arsène MBABEH MEYE", layout="wide")
st.title("Développé par Arsène MBABEH MEYE - Application de Data Science")

# ─────── Objectif métier ───────
st.markdown("### 0. Quel est votre objectif métier ?")
objectif = st.radio("Choisissez une intention parmi les suivantes :", [
    "Prédire un phénomène futur (ex: coupure réseau)",
    "Comprendre les causes d’un phénomène (ex: réclamations)",
    "Explorer les données sans prédiction"
])

if objectif == "Explorer les données sans prédiction":
    st.info("Vous pouvez explorer vos données dans la section suivante.")

# ───────section pour charger les fichiers ───────
st.markdown("### 1. Importation des données")
uploaded_file = st.file_uploader("Choisir un fichier CSV", type=["csv"])

if uploaded_file is not None:
    sep = st.selectbox("Choisir le séparateur du fichier", [",", ";", "\t"], index=1)
    try:
        df = pd.read_csv(uploaded_file, sep=sep)
        st.success("Fichier chargé avec succès")
        st.dataframe(df.head())
    except Exception as e:
        st.error(f"Erreur lors du chargement : {e}")
        st.stop()
else:
    st.stop()

# ─────── Analyse exploratoire ───────
st.markdown("### 2. Analyse exploratoire")

if st.checkbox("Afficher les statistiques descriptives"):
    st.write(df.describe())

if st.checkbox("Afficher les valeurs manquantes"):
    st.dataframe(df.isnull().sum().reset_index().rename(columns={0: 'Nombre de valeurs manquantes', 'index': 'Colonnes'}))

if st.checkbox("Afficher les corrélations (variables numériques uniquement)"):
    corr = df.select_dtypes(include=np.number).corr()
    fig_corr, ax_corr = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax_corr)
    st.pyplot(fig_corr)

if st.checkbox("Afficher la distribution des variables numériques"):
    numeric_cols = df.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        fig, ax = plt.subplots()
        sns.histplot(df[col].dropna(), kde=True, ax=ax)
        ax.set_title(f"Distribution de {col}")
        st.pyplot(fig)

if st.checkbox("Afficher les variables catégorielles (fréquence)"):
    cat_cols = df.select_dtypes(include='object').columns
    for col in cat_cols:
        st.write(f"#### Fréquence des valeurs pour '{col}'")
        st.dataframe(df[col].value_counts())

st.markdown("#### Visualisation interactive")
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
    st.warning(f"Erreur d'affichage : {e}")

# ─────── Prédiction ? ───────
st.markdown("### 3. Souhaitez-vous réaliser une prédiction ?")
predict_now = st.radio("Choisir", ["Non", "Oui"])

if predict_now == "Oui":
    st.markdown("### 4. Configuration du modèle")
    target = st.selectbox("Choisir la variable cible à prédire", df.columns)

    if target:
        X = df.drop(columns=[target])
        y = df[target]

        X = pd.get_dummies(X)
        df_clean = pd.concat([X, y], axis=1).dropna()
        X = df_clean.drop(columns=[target])
        y = df_clean[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        is_classification = y.dtype == 'object' or y.nunique() < 20

        model_options = {
            "Random Forest": RandomForestClassifier() if is_classification else RandomForestRegressor(),
            "SVM": SVC() if is_classification else SVR(),
            "KNN": KNeighborsClassifier() if is_classification else KNeighborsRegressor(),
            "Régression": LogisticRegression() if is_classification else LinearRegression()
        }

        if xgb_installed:
            model_options["XGBoost"] = XGBClassifier() if is_classification else XGBRegressor()

        model_name = st.selectbox("Choisir un modèle", list(model_options.keys()))

        # Explication du modèle
        explanations = {
            "Random Forest": "Forêt d'arbres de décision, robuste, bon pour les données tabulaires.",
            "SVM": "Sépare les classes en maximisant les marges. Efficace sur petits jeux.",
            "KNN": "Compare chaque observation à ses k plus proches voisins.",
            "Régression": "Modèle linéaire simple pour établir une relation entre variables.",
            "XGBoost": "Modèle puissant de boosting, très performant sur grands jeux de données."
        }
        st.info(f"\n**{model_name} :** {explanations[model_name]}")

        model = model_options[model_name]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # ─────── Interprétation ───────
        st.markdown("### 5. Résultats et interprétation")

        if is_classification:
            acc = accuracy_score(y_test, y_pred)
            st.write(f"Précision du modèle : {acc:.2f}")
            st.text(classification_report(y_test, y_pred))

            cm = confusion_matrix(y_test, y_pred)
            fig_cm, ax_cm = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
            ax_cm.set_xlabel("Prédit")
            ax_cm.set_ylabel("Réel")
            st.pyplot(fig_cm)

        else:
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            st.write(f"Erreur quadratique moyenne (RMSE) : {rmse:.2f}")
            st.info("\nLe RMSE indique l'écart moyen entre les valeurs réelles et prédites. Plus il est faible, plus le modèle est précis.")

        if hasattr(model, "feature_importances_"):
            st.markdown("#### Importance des variables")
            importances = pd.DataFrame({
                "Variable": X.columns,
                "Importance": model.feature_importances_
            }).sort_values(by="Importance", ascending=False).head(10)

            st.dataframe(importances)
            fig_imp, ax_imp = plt.subplots()
            sns.barplot(x="Importance", y="Variable", data=importances, ax=ax_imp)
            st.pyplot(fig_imp)

        # ─────── Export ───────
        result_df = X_test.copy()
        result_df["Valeur réelle"] = y_test
        result_df["Prédiction"] = y_pred

        csv = result_df.to_csv(index=False).encode("utf-8")
        st.download_button("Télécharger les prédictions", data=csv, file_name="predictions.csv", mime="text/csv")

        if st.button("Sauvegarder le modèle"):
            joblib.dump(model, f"modele_{model_name}.pkl")
            st.success("Modèle sauvegardé")
