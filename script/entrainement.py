import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from utiles import *

df = pd.read_csv(r"C:\Users\Lenovo\Desktop\Mailing_project\data\DataSetUtile.csv")
data_cleaned = pipeline(df, "text")

#   Vectoriser le texte stemmé
vectorizer = TfidfVectorizer(max_features=1000)

data_cleaned['stems_joined'] = data_cleaned['stems'].apply(lambda tokens: ' '.join(tokens))

X = vectorizer.fit_transform(data_cleaned['stems_joined'])
y = data_cleaned["label"]
joblib.dump(vectorizer, r"C:\Users\Lenovo\Desktop\Mailing_project\models\vectorizer.pkl")
print("Vectorizer sauvegardé avec succès !")


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify= y
)

# Liste des modèles avec leurs grilles de recherche
model_configs = {
    "Decision Tree": {
        "model": DecisionTreeClassifier(random_state=42),
        "params": {
            "max_depth": [5, 10, None],
            "criterion": ["gini", "entropy"]
        }
    },
    "Naive Bayes": {
        "model": MultinomialNB(),
        "params": {
            "alpha": [0.5, 1.0, 1.5]
        }
    },
    "SVM": {
        "model": SVC(),
        "params": {
            "kernel": ["linear", "rbf"],
            "C": [0.1, 1, 10]
        }
    }
}

#   Liste des differents resultats
results = []

for name, config in model_configs.items():
    print(f"\n🔍 GridSearch pour {name}...")
    
    grid = GridSearchCV(config["model"], config["params"], cv=5, scoring='f1_macro', n_jobs=-1)
    grid.fit(X_train, y_train)
    
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    
    # Stocker les résultats principaux
    results.append({
        "Modèle": name,
        "F1_macro (val)": grid.best_score_,
        "F1_macro (test)": report["macro avg"]["f1-score"],
        "Précision (test)": report["macro avg"]["precision"],
        "Rappel (test)": report["macro avg"]["recall"],
        "Meilleurs paramètres": grid.best_params_
    })

df_results = pd.DataFrame(results)
df_results = df_results.sort_values(by="F1_macro (test)", ascending=False)
df_results.to_csv(r"C:\Users\Lenovo\Desktop\Mailing_project\metrics\resultats.csv")

best_params = grid.best_params_
print(best_params)
# Recréer un modèle avec les meilleurs paramètres
mon_modele_optimal = SVC(**best_params)

# Entraînement sur tout X_train (tu peux aussi le faire sur tout X si tu veux déployer)
mon_modele_optimal.fit(X_train, y_train)

# Faire des prédictions sur le test set
y_pred = mon_modele_optimal.predict(X_test)

# Évaluer le modèle

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

#   Sauvegarde de mon modele
joblib.dump(mon_modele_optimal, r"C:\Users\Lenovo\Desktop\Mailing_project\models\meilleur_modele_svm.pkl")
print("Modele Sauvegarde avec Succces")