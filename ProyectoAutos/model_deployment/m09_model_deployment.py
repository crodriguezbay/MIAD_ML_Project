#!/usr/bin/python

import pandas as pd
import joblib
import sys
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler

def predict_proba(year,mileage,state,make,model):

    Reg = joblib.load(os.path.dirname(__file__) + '/proyecto_reg.pkl') 

    # Create features -----------------------------------------------------------------------------------------
    X_test = pd.DataFrame([[year,mileage,' '+state,make,model]], columns=['Year','Mileage','State','Make','Model'])

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
    preprocess.fit_transform(X)

    print("Muestra original")
    print(X_test.head())
    X_pred = preprocess.transform(X_test)
    print("Muestra transformada")
    print(X_pred)
    
    # Make prediction
    y_pred = Reg.predict(X_pred)
    print("Predicción del precio")
    print(y_pred)

    return y_pred[0]


if __name__ == "__main__":
    
    if len(sys.argv) == 1:
        print('Agrega los parámetros necesarios')
    else:

        year = sys.argv[1]
        mileage = sys.argv[2]
        state = sys.argv[3]
        make = sys.argv[4]
        model = sys.argv[5]

        p1 = predict_proba(year,mileage,state,make,model)

        print(year)
        print(mileage)
        print(state)
        print(make)
        print(model)
        print('Precio del automóvil: ', p1)
        