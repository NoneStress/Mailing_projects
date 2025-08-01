import pandas as pd
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

# Télécharger les ressources nécessaires
# nltk.download('punkt')
# nltk.download('stopwords')

# Initialiser le stemmer et les stopwords
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# 1. Mise en minuscules et suppression des lignes vides dans une colonne
def convertisseur(df, column):
    df = df.dropna(subset=[column])
    df[column] = df[column].str.lower()
    return df

# 2. Suppression des doublons dans une colonne
def dropdoublons(df, column):
    return df.drop_duplicates(subset=column, keep="first")

# 3. Nettoyage de texte (ponctuation, caractères spéciaux)
def nettoyer_ponctuation(text):
    if pd.isna(text):
        return ""
    return re.sub(r'[^a-zA-Z0-9\s]', '', str(text))

# 4. Tokenisation
def tokenize_cell(text):
    if pd.isna(text):
        return []
    return word_tokenize(str(text))

# 5. Suppression des stopwords
def supprimer_stopwords(tokens):
    return [token for token in tokens if token.lower() not in stop_words]

# 6. Stemming
def appliquer_stemming(tokens):
    return [stemmer.stem(token) for token in tokens]

# 7. Pipeline complète
def pipeline(df, column):
    df = convertisseur(df, column)
    print("*****************Conversion effectuee*****************")

    df = dropdoublons(df, column)
    print("************(*****Doublons supprimes******************")

    df["text_clean"] = df[column].apply(nettoyer_ponctuation)
    print("*****************Ponctuation retiree******************")

    df["tokens"] = df["text_clean"].apply(tokenize_cell)
    print("***********(********Texte tokenise********************")

    df["tokens_no_stop"] = df["tokens"].apply(supprimer_stopwords)
    print("******************Stop Words retires******************")

    df["stems"] = df["tokens_no_stop"].apply(appliquer_stemming)
    print("****************Application du Stemmin****************")
    
    return df
