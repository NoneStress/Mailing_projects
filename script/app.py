import streamlit as st
import joblib
import pandas as pd
from utiles import *  # Ton pipeline custom

#   Chargement du modèle et du vectorizer 
model = joblib.load(r"C:\Users\Lenovo\Desktop\Mailing_project\models\meilleur_modele_svm.pkl")
vectorizer = joblib.load(r"C:\Users\Lenovo\Desktop\Mailing_project\models\vectorizer.pkl")

#   Interface utilisateur
st.set_page_config(page_title="Détection de Spam", layout="centered")
st.title("📧 Détecteur de Spam")
st.markdown("Entrez un message ci-dessous et l'application vous dira s'il est **spam** ou **légitime**.")

#   Zone de texte
user_input = st.text_area("✉️ Message à analyser :", "")

#   Action bouton
if st.button("Analyser"):
    if not user_input.strip():
        st.warning("Le champ est vide. Veuillez entrer un message.")
    else:
        # Créer un mini DataFrame pour passer par le pipeline
        temp_df = pd.DataFrame([user_input], columns=["text"])
        temp_df = pipeline(temp_df, "text")
        temp_df["stems_joined"] = temp_df["stems"].apply(lambda x: " ".join(x))
        
        # Vectorisation
        X_input = vectorizer.transform(temp_df["stems_joined"])

        # Prédiction
        prediction = model.predict(X_input)[0]
        score = model.decision_function(X_input)[0]  # pour SVM

        # Résultat
        label = " ✅ Ham " if prediction == 0 else " 🚨 Spam "
        st.subheader("Résultat :")
        st.success(label)
        st.caption(f"Score de décision SVM : {score:.4f}")
