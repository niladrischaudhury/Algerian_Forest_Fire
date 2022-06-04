from flask import Flask, request, jsonify, render_template
from flask_cors import CORS, cross_origin
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

reg_model = pickle.load(open('model.pkl', 'rb'))

@app.route('/', methods=['GET'])  # route to display the home page
@cross_origin()
def homePage():
    print("Starting home page")
    return render_template("index.html")

@app.route('/predict_temp', methods=['POST'])
@cross_origin()
def predict_temp_api():
    #day = request.json['day']
    #month = request.json['month']
    #year = request.json['year']
    #RH = request.json['RH']
    #WS = request.json['WS']
    #Rain = request.json['Rain']
    #FFMC = request.json['FFMC']
    #DMC = request.json['DMC']
    #DC = request.json['DC']
    #ISI = request.json['ISI']
    #BUI = request.json['BUI']
    #FWI = request.json['FWI']

    #model_request = [[day, month, year, RH, WS, Rain, FFMC, DMC, DC, ISI, BUI, FWI]]
    #print('Request:', model_request)
    #return jsonify(str(response_message))

    data = [float(x) for x in request.form.values()]
    final_feature = [np.array(data)]
    print('Request:', final_feature)
    model_predict = reg_model.predict(final_feature)[0]
    print("Predicted temp: ", model_predict)
    response_message = model_predict
    return render_template('index.html', prediction_text = "Predicted temperature is {}".format(response_message))

if __name__ == '__main__':
    app.run()
