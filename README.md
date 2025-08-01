# 📧 Détection de Spam dans les Emails - Projet NLP

Ce projet a pour objectif de détecter si un email est un **spam** ou un **ham** (email légitime) à l’aide d’algorithmes de machine learning et de techniques de traitement automatique du langage naturel (NLP). Une application Streamlit permet à l’utilisateur de tester ses propres messages.

---

## 🚀 Fonctionnalités

- Nettoyage de texte (ponctuation, minuscules, tokenisation, stopwords, stemming)
- Vectorisation TF-IDF des textes
- Entraînement de plusieurs modèles : SVM, Naive Bayes, Decision Tree
- Optimisation avec GridSearchCV
- Sauvegarde du meilleur modèle et du vectorizer
- Analyse exploratoire (EDA) avec visualisations
- Interface interactive Streamlit pour prédire en temps réel

---

## 📁 Structure du projet

```
Mailing_project/
├── app.py                      
├── data/
│   └── DataSetUtile.csv        
├── figures/                    
│   ├── Distribution_des_mails.png
│   ├── Matrice_de_corrélation.png
│   ├── Nuage_de_mots_HAM.png
│   ├── Nuage_de_mots_SPAM.png
│   ├── Violin_Plot.png
│   └── Volume_de_hams_dans_le_temps.png
├── metrics/
│   └── resultats.csv           
├── models/
│   ├── meilleur_modele_svm.pkl 
│   └── vectorizer.pkl          
├── scripts/
│   ├── utiles.py               
│   ├── entrainement.py         
│   └── EDA.ipynb                  
├── README.md
└── requirements.txt
```

---

## ▶️ Utilisation de l'application

### 1. Installation des dépendances

```bash
pip install -r requirements.txt
```

### 2. Lancer l'application

```bash
streamlit run app.py
```

---

## 📊 Résultats

- Les **graphiques d’analyse exploratoire** sont enregistrés dans le dossier `figures/`.
- Les **résultats de l’évaluation des modèles** sont enregistrés dans `metrics/resultats.csv`.

---

## 📦 Environnement requis

- `pandas`
- `numpy`
- `scikit-learn`
- `nltk`
- `streamlit`
- `joblib`
- `matplotlib`, `seaborn`

> N'oubliez pas d'initialiser les ressources NLTK une fois :
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')


Projet développé par **Nasser Yerbanga**
