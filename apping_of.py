# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 11:54:15 2020

@author: Sunshine
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 19:59:56 2020

@author: Sunshine
"""


import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('internshala.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index2.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    return render_template('index2.html', prediction_text='Campus Recruitment : {}'.format(prediction))


if __name__ == "__main__":
    app.run(debug=True)