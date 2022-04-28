#!/usr/bin/python
from flask import Flask
from flask_restplus import Api, Resource, fields
import joblib
from m09_model_deployment import predict_proba
import logging

app = Flask(__name__)

api = Api(
    app, 
    version='1.0', 
    title='Predicción de precios de vehículos usados',
    description='API que predice el precio de un vehiculo usado utilizando variables como: año, marca, modelo, entre otras')

ns = api.namespace('predict', 
     description='Predicción de precios de vehículos usados')
   
parser = api.parser()

parser.add_argument(
    'URL', 
    type=str, 
    required=True, 
    help='URL to be analyzed', 
    location='args')
    
parser.add_argument(
    'Year', 
    type=int, 
    required=True, 
    help='Año', 
    location='args')

parser.add_argument(
    'Mileage', 
    type=float, 
    required=True, 
    help='Kilometraje', 
    location='args')

parser.add_argument(
    'State', 
    type=str, 
    required=True, 
    help='Estado de los EE.UU.', 
    location='args')

parser.add_argument(
    'Make', 
    type=str, 
    required=True, 
    help='Marca', 
    location='args')

parser.add_argument(
    'Model', 
    type=str, 
    required=True, 
    help='Modelo', 
    location='args')

resource_fields = api.model('Resource', {
    'result': fields.String,
})

@ns.route('/')
class PhishingApi(Resource):

    @api.doc(parser=parser)
    @api.marshal_with(resource_fields)
    def get(self):
        args = parser.parse_args()
        app.logger.info(args)
        
        return {
         "result": predict_proba(args['URL'])
        }, 200
    
    
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=8888)
