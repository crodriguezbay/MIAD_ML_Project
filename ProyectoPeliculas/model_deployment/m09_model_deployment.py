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

def predict_proba(plot):

    #Reg = joblib.load(os.path.dirname(__file__) + '/proyecto_reg.pkl') 

    # Create features -----------------------------------------------------------------------------------------
    X_test_data = pd.DataFrame([[plot]], columns=['plot'])

    print("Muestra original---------------------------------")
    print(X_test_data.head())

    #Preprocesamiento de los datos -----------------------------------------------------------------------------------

    #Limpieza
    X_test_data['clean_plot'] = X_test_data['plot'].apply(lambda x: clean_text(x))
    print("Después de Limpieza---------------------------------")
    print(X_test_data.head())

    #Remove stopwords
    X_test_data['clean_plot'] = X_test_data['clean_plot'].apply(lambda x: remove_stopwords(x))
    print("Después de stopwords---------------------------------")
    print(X_test_data.head())

    #X_pred = preprocess.transform(X_test_data)
    #print("Muestra transformada")
    #print(X_pred)
    
    # Make prediction
    #y_pred = Reg.predict(X_pred)
    #print("Predicción del precio")
    #print(y_pred)

    return 'Gender'


if __name__ == "__main__":
    
    if len(sys.argv) == 1:
        print('Agrega los parámetros necesarios')
    else:
        plot = sys.argv[1]
        p1 = predict_proba(plot)
        print(plot)
        print('Género de la película', p1)
        