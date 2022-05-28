#!/usr/bin/python
from flask import Flask
from flask_restplus import Api, Resource, fields
import joblib
from m09_model_deployment import predict_proba
import logging
import json

app = Flask(__name__)

api = Api(
    app, 
    version='1.0', 
    title='Clasificación de género de películas',
    description='API que predice el género de películas')

ns = api.namespace('predict', 
     description='Clasificación de género de películas')
   
parser = api.parser()

parser.add_argument(
    'Plot', 
    type=str, 
    required=True, 
    help='Trama de la película', 
    location='args')

resource_fields = api.model('Resource', {
    'result': fields.String,
})

@ns.route('/')
class GenderApi(Resource):

    @api.doc(parser=parser)
    @api.marshal_with(resource_fields)
    def get(self):
        args = parser.parse_args()
        app.logger.info(args)

        json_object = json.loads(predict_proba(args['Plot']))

        json_data_list = [{'p_Action' : json_object['p_Action']},
                        {'p_Adventure' : json_object['p_Adventure']}
                    ]

        return {
         'result' : json_data_list
        }, 200
    
    
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=8888)
