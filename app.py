from flask import Flask, request, jsonify, app
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

reg_model = pickle.load(open('model.pkl', 'rb'))

@app.route('/predict_temp', methods=['POST'])
def predict_temp_api():
    day = request.json['day']
    month = request.json['month']
    year = request.json['year']
    RH = request.json['RH']
    WS = request.json['WS']
    Rain = request.json['Rain']
    FFMC = request.json['FFMC']
    DMC = request.json['DMC']
    DC = request.json['DC']
    ISI = request.json['ISI']
    BUI = request.json['BUI']
    FWI = request.json['FWI']

    model_request = [[day, month, year, RH, WS, Rain, FFMC, DMC, DC, ISI, BUI, FWI]]
    print('Request:', model_request)
    model_predict = reg_model.predict(model_request)[0]
    response_message = "Predicted temperature is: " + str(model_predict)

    return jsonify(str(response_message))


if __name__ == '__main__':
    app.run(debug=True)
