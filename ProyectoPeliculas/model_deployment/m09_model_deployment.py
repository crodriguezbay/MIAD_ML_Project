#!/usr/bin/python

import pandas as pd
import joblib
import sys
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from libs import clean_text,remove_stopwords
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

def predict_proba(plot):

    clf = joblib.load(os.path.dirname(__file__) + '/clf.pkl') 
    #vect1 = joblib.load(os.path.dirname(__file__) + '/vect1.pkl') 
    #vect_lemas = joblib.load(os.path.dirname(__file__) + '/vect_lemas.pkl') 

    # Create features -----------------------------------------------------------------------------------------
    X_test_data = pd.DataFrame([[plot]], columns=['plot'])

    print("Muestra original---------------------------------")
    print(X_test_data.head())

    #Preprocesamiento de los datos -----------------------------------------------------------------------------------

    #Datos de train necesarios para limpiar y normalizar bajo los mismos parametros
    dataTraining = pd.read_csv('https://github.com/albahnsen/MIAD_ML_and_NLP/raw/main/datasets/dataTraining.zip', encoding='UTF-8', index_col=0)
    #Limpieza
    dataTraining['clean_plot'] = dataTraining['plot'].apply(lambda x: clean_text(x))
    #Remove stopwords
    dataTraining['clean_plot'] = dataTraining['clean_plot'].apply(lambda x: remove_stopwords(x))
    #Lematización
    vect = TfidfVectorizer(min_df=3)
    X_dtm = vect.fit_transform(dataTraining['clean_plot'])

    vect1 = TfidfVectorizer()
    vect1.fit(dataTraining['clean_plot'])
    words = list(vect1.vocabulary_.keys())[:100]
    wordnet_lemmatizer = WordNetLemmatizer()
    
    def split_into_lemmas(text):
        text = text.lower()
        words = text.split()
        return [wordnet_lemmatizer.lemmatize(word) for word in words]
    
    vect_lemas = TfidfVectorizer(analyzer=split_into_lemmas)
    X_train_l = vect_lemas.fit_transform(dataTraining['clean_plot'])

    #DATOS NUEVOS--------------------------------------------------------------------
    #Limpieza
    X_test_data['clean_plot'] = X_test_data['plot'].apply(lambda x: clean_text(x))
    print("Después de Limpieza---------------------------------")
    print(X_test_data.head())

    #Remove stopwords
    X_test_data['clean_plot'] = X_test_data['clean_plot'].apply(lambda x: remove_stopwords(x))
    print("Después de stopwords---------------------------------")
    print(X_test_data.head())

    #Lematización
    X_test_dtm = vect_lemas.transform(X_test_data['clean_plot'])
    print("Muestra transformada---------------------------------")
    print(X_test_dtm)
    
    # Predicción de la muestra de test
    y_pred_test_genres = clf.predict_proba(X_test_dtm)
    print("Predicción del género")
    print(y_pred_test_genres)

    cols = ['p_Action', 'p_Adventure', 'p_Animation', 'p_Biography', 'p_Comedy', 'p_Crime', 'p_Documentary', 'p_Drama', 'p_Family',
        'p_Fantasy', 'p_Film-Noir', 'p_History', 'p_Horror', 'p_Music', 'p_Musical', 'p_Mystery', 'p_News', 'p_Romance',
        'p_Sci-Fi', 'p_Short', 'p_Sport', 'p_Thriller', 'p_War', 'p_Western']
    
    res = pd.DataFrame(y_pred_test_genres, index=X_test_data.index, columns=cols)

    print(res)

    return 'Gender'


if __name__ == "__main__":
    
    if len(sys.argv) == 1:
        print('Agrega los parámetros necesarios')
    else:
        plot = sys.argv[1]
        p1 = predict_proba(plot)
        print(plot)
        print('Género de la película', p1)
        