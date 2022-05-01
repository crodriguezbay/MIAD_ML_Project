#!/usr/bin/python

import pandas as pd
import joblib
import sys
import os
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler

def predict_proba(url,year,mileage,state,make,model):

    clf = joblib.load(os.path.dirname(__file__) + '/phishing_clf.pkl') 
    Reg = joblib.load(os.path.dirname(__file__) + '/proyecto_clf.pkl') 

    url_ = pd.DataFrame([url], columns=['url'])

    print(url)

  
    # Create features
    keywords = ['https', 'login', '.php', '.html', '@', 'sign']
    for keyword in keywords:
        url_['keyword_' + keyword] = url_.url.str.contains(keyword).astype(int)

    url_['lenght'] = url_.url.str.len() - 2
    domain = url_.url.str.split('/', expand=True).iloc[:, 2]
    url_['lenght_domain'] = domain.str.len()
    url_['isIP'] = (url_.url.str.replace('.', '') * 1).str.isnumeric().astype(int)
    url_['count_com'] = url_.url.str.count('com')

    print(url_.head())

    # Create features -----------------------------------------------------------------------------------------
    X_test = pd.DataFrame([[year,mileage,state,make,model]], columns=['Year','Mileage','State','Make','Model'])

    #Preprocesamiento de los datos -----------------------------------------------------------------------------------

    # Se necesita una transformación de las variables categoricas para poder implementar 
    # el modelo de regresión
    dataTraining = pd.read_csv('https://raw.githubusercontent.com/albahnsen/MIAD_ML_and_NLP/main/datasets/dataTrain_carListings.zip')
    X = dataTraining.drop(['Price'],axis=1)

    # Identificación de las variables categoricas
    categorical_mask = (X.dtypes == 'object')

    # Creación de una lista con los nombres de las columnas categoricas
    categorical_columns = X.columns[categorical_mask].tolist()

    # Creación de lista con las variables unicas
    unique_list = [X[c].unique().tolist() for c in categorical_columns]

    # Creación del OneHotEncoder
    ohe = OneHotEncoder(categories=unique_list)

    # Crear el objeto para el preprocesamiento del OneHotEncoder, adicional se realiza una estandarización
    # de la columna 'Mileage'
    preprocess = make_column_transformer(
        (StandardScaler(), ['Mileage']),
        (ohe, categorical_columns),
        ('passthrough',  categorical_mask[~categorical_mask].index.tolist()))

    print(X.head())
    print(X_test.head())
    X_pred = preprocess.fit_transform(X_test)
    print(X_pred)
    

    y_pred = Reg.predict(X_pred)
    print(y_pred)

    # Make prediction
    p1 = clf.predict_proba(url_.drop('url', axis=1))[0,1]

    return p1


if __name__ == "__main__":
    
    if len(sys.argv) == 1:
        print('Please add an URL')
        
    else:

        url = sys.argv[1]

        p1 = predict_proba(url)
        
        print(url)
        print('Probability of Phishing: ', p1)
        