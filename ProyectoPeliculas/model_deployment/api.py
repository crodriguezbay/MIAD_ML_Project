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

        return {
         'result' : {'p_Action' : json_object['p_Action'],
                    'p_Adventure' : json_object['p_Adventure'], 
                    'p_Animation' : json_object['p_Animation'], 
                    'p_Biography' : json_object['p_Biography'], 
                    'p_Comedy' : json_object['p_Comedy'], 
                    'p_Crime' : json_object['p_Crime'], 
                    'p_Documentary' : json_object['p_Documentary'], 
                    'p_Drama' : json_object['p_Drama'], 
                    'p_Family' : json_object['p_Family'],
                    'p_Fantasy' : json_object['p_Fantasy'], 
                    'p_Film-Noir' : json_object['p_Film-Noir'], 
                    'p_History' : json_object['p_History'], 
                    'p_Horror' : json_object['p_Horror'], 
                    'p_Music' : json_object['p_Music'], 
                    'p_Musical' : json_object['p_Musical'], 
                    'p_Mystery' : json_object['p_Mystery'], 
                    'p_News' : json_object['p_News'], 
                    'p_Romance' : json_object['p_Romance'],
                    'p_Sci-Fi' : json_object['p_Sci-Fi'], 
                    'p_Short' : json_object['p_Short'], 
                    'p_Sport' : json_object['p_Sport'], 
                    'p_Thriller' : json_object['p_Thriller'], 
                    'p_War' : json_object['p_War'], 
                    'p_Western' : json_object['p_Western']
         }
        }, 200
    
    
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=8888)
