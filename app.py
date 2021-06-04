#import dependencies
from flask import Flask, render_template, redirect, request
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from pickle import dump, load
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#creating an app
app = Flask(__name__)

diabetes_model = load(open('diabetes_model.pkl', 'rb'))
diabetes_scaler = load(open('diabetes_scaler.pkl', 'rb'))
#Homepage
@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/diagnose', methods = ['POST', 'GET'])
def diagnose():
    if request.method == 'POST':
        form_data = request.form
        #Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age
        result  = bool(diabetes_model.predict(diabetes_scaler.transform([[request.form['Pregnancies'],request.form['Glucose'],\
            request.form['BloodPressure'],request.form['SkinThickness'],request.form['Insulin'],request.form['BMI'],request.form['DiabetesPedigreeFunction'],request.form['Age']]])))

        return render_template('diagnose.html', result=result)
    return render_template('diagnose.html')

@app.route('/howitworks')
def howthisworks():
    return render_template('howitworks.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')


if __name__ == "__main__":
    app.run(debug=True)
