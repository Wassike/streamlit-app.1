import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import math

# --- Configuration de la page ---
st.set_page_config(page_title="Prévision des ventes avec LSTM", layout="wide")
st.title("📊 Application de Prévision des Ventes avec LSTM")

# --- Chargement du fichier CSV ---
uploaded_file = st.file_uploader("📂 Importer un fichier CSV contenant les ventes (une colonne numérique)", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.subheader("Aperçu des données")
    st.dataframe(data.head())

    # --- Vérification du contenu ---
    numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
    if not numeric_cols:
        st.error("Aucune colonne numérique détectée. Merci d’importer un fichier avec des valeurs de ventes.")
    else:
        target_col = st.selectbox("📈 Sélectionner la colonne cible :", numeric_cols)

        # --- Normalisation ---
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data[[target_col]])

        # --- Création des séquences ---
        time_step = st.slider("🕒 Taille de la fenêtre temporelle", 5, 60, 20)
        X, y = [], []
        for i in range(time_step, len(scaled_data)):
            X.append(scaled_data[i-time_step:i, 0])
            y.append(scaled_data[i, 0])
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        # --- Paramètres d'entraînement ---
        st.sidebar.header("⚙️ Paramètres du modèle")
        epochs = st.sidebar.slider("Nombre d'epochs", 10, 200, 50)
        batch_size = st.sidebar.slider("Taille du batch", 8, 64, 16)

        # --- Définition du modèle ---
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
        model.add(LSTM(50))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')

        # --- Bouton d'entraînement ---
        if st.button("🔁 Entraîner le modèle"):
            with st.spinner("Entraînement du modèle en cours..."):
                model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)
            st.success("✅ Modèle entraîné avec succès !")

            # --- Prédiction ---
            prediction = model.predict(X)
            prediction = scaler.inverse_transform(prediction)
            original = scaler.inverse_transform(y.reshape(-1, 1))

            # --- Évaluation ---
            rmse = math.sqrt(mean_squared_error(original, prediction))
            st.metric(label="Erreur RMSE", value=f"{rmse:.2f}")

            # --- Graphique ---
            st.subheader("📉 Comparaison entre les valeurs réelles et les prédictions")
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(original, label="Valeurs réelles")
            ax.plot(prediction, label="Prédictions LSTM", linestyle='dashed')
            ax.legend()
            ax.set_xlabel("Temps")
            ax.set_ylabel("Ventes")
            st.pyplot(fig)

else:
    st.info("Veuillez importer un fichier CSV pour commencer.")