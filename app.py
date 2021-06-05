#import dependencies
from flask import Flask, render_template, redirect, request
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from pickle import dump, load
#from tensorflow import lite
#from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
#import tensorflow as tf


#creating an app
app = Flask(__name__)

diabetes_model = load(open('diabetes_model.pkl', 'rb'))
diabetes_scaler = load(open('diabetes_scaler.pkl', 'rb'))
#retinopathy_model = load_model("Retinopathy_model_trained_20-40-40.h5")

#Homepage
@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/diagnose', methods = ['POST', 'GET'])
def diagnose():
    diabetes_result = 2
    retinopathy_result = 2
    if request.method == 'POST':
        form_data = request.form
        #Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age
        diabetes_result  = diabetes_model.predict(diabetes_scaler.transform([[form_data['Pregnancies'],form_data['Glucose'],\
            form_data['BloodPressure'],form_data['SkinThickness'],form_data['Insulin'],form_data['BMI'],\
            form_data['DiabetesPedigreeFunction'],form_data['Age']]]))[0]
        #select the images
        #retinopathy_result = retinopathy_model.predict_classes(form_data)
        return render_template('diagnose.html',diabetes_result=diabetes_result,age1 = form_data['Age'], 
        bmi1 = form_data['BMI'],skin1 = form_data['SkinThickness'], ins1 = form_data['Insulin'],
        glu1 = form_data['Glucose'], bp1 = form_data['BloodPressure'], dpf1 = form_data['DiabetesPedigreeFunction'],
        preg1 = form_data['Pregnancies'] )
        #, retinopathy_result=retinopathy_result)
    return render_template('diagnose.html',diabetes_result=diabetes_result)
    
@app.route('/howitworks')
def howthisworks():
    return render_template('howitworks.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')


if __name__ == "__main__":
    app.run(debug=True)
