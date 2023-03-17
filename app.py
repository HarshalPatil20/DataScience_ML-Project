import pickle
from flask import Flask, request, app, jsonify, url_for, render_template, redirect, flash, session, escape
from xgboost import XGBRegressor
import numpy as np
import pandas as pd

app=Flask(__name__)
#load model
regmodel=pickle.load(open('regmodel.pkl','rb'))
#scalar=pickle.load(open())
@app.route('/')


def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])

def predict_api():
    data=request.json['data']
    print(data)
    #input_data = np.array(list(data.values())).reshape(1,-1)
    input_data = np.array(list(data.values())).reshape(1,-1)
    #print(input_data)
    output = regmodel.predict(input_data)
    print(output[0])
    return jsonify(output[0])

if __name__=='__main__':
    app.run(debug=True)