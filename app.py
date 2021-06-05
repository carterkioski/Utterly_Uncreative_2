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
import pandas as pd
import random, os
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

#import tensorflow as tf


#creating an app
app = Flask(__name__)

diabetes_model = load(open('diabetes_model.pkl', 'rb'))
diabetes_scaler = load(open('diabetes_scaler.pkl', 'rb'))
retinopathy_model = load_model("Code/Models/Retinopathy_model_trained_20-40-40.h5")

# setup for the retinal model
Image_info_df = pd.read_csv('Data/Sample_Data/train.csv')

diagnosis_dict_binary = {
    0: 'No_DR',
    1: 'DR',
    2: 'DR',
    3: 'DR',
    4: 'DR'
}

diagnosis_dict = {
    0: 'No_DR',
    1: 'Mild',
    2: 'Moderate',
    3: 'Severe',
    4: 'Proliferate_DR',
}
Image_info_df['binary_type'] =  Image_info_df['diagnosis'].map(diagnosis_dict_binary.get)
Image_info_df['type'] = Image_info_df['diagnosis'].map(diagnosis_dict.get)


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
        special = [190]         
        special_df=Image_info_df.loc[special]
        # Create working directories for the image_file
        base_dir = ''

        special_dir = os.path.join(base_dir, 'Special')

        if os.path.exists(base_dir):
            shutil.rmtree(base_dir)

        if os.path.exists(special_dir):
            shutil.rmtree(special_dir)
        os.makedirs(special_dir)
        # # Copy images to respective working directory
        src_dir = 'Data/Sample_Images/'
        for index, row in special_df.iterrows():
            diagnosis = row['type']
            binary_diagnosis = row['binary_type']
            id_code = row['id_code'] + ".png"
            srcfile = os.path.join(src_dir, diagnosis, id_code)
            dstfile = os.path.join(special_dir, diagnosis)
            os.makedirs(dstfile, exist_ok = True)
            shutil.copy(srcfile, dstfile)
            special_path = 'Special'

        Diagnosis = ImageDataGenerator(rescale = 1./255).flow_from_directory(special_path, target_size=(224,224), shuffle = True)

        predicted = retinopathy_model.predict(Diagnosis)
        predicted = np.argmax(predicted,axis=1)

        # Map the label
        # labels = (Diagnosis.class_indices)
        # labels = dict((v,k) for k,v in labels.items())
        Labels = {0: 'Mild', 1: 'Moderate', 2: 'No_DR', 3: 'Proliferate_DR', 4: 'Severe'}
        predicted = [Labels[k] for k in predicted]
    
        #retinopathy_result = retinopathy_model.predict_classes(form_data)
        
        
        
        
        
        return render_template('diagnose.html',diabetes_result=diabetes_result,age1 = form_data['Age'], 
        bmi1 = form_data['BMI'],skin1 = form_data['SkinThickness'], ins1 = form_data['Insulin'],
        glu1 = form_data['Glucose'], bp1 = form_data['BloodPressure'], dpf1 = form_data['DiabetesPedigreeFunction'],
        preg1 = form_data['Pregnancies'],retinopathy_result=predicted[:1][0])
    return render_template('diagnose.html',diabetes_result=diabetes_result,retinopathy_result=1)
    
@app.route('/howitworks')
def howthisworks():
    return render_template('howitworks.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')


if __name__ == "__main__":
    app.run(debug=True)
