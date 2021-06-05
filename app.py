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
import random, os
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
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

#Homepage
@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/diagnose', methods = ['POST', 'GET'])
def diagnose():
    diabetes_result = 2
    retinopathy_result = 2
    if request.method == 'POST':
        special_path = 'Special'
        form_data = request.form
        #Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age
        diabetes_result  = diabetes_model.predict(diabetes_scaler.transform([[form_data['Pregnancies'],form_data['Glucose'],\
            form_data['BloodPressure'],form_data['SkinThickness'],form_data['Insulin'],form_data['BMI'],\
            form_data['DiabetesPedigreeFunction'],form_data['Age']]]))[0]
        #select the images
        special = [int(form_data['retina'])]
        special_df=Image_info_df.loc[special]
        # Create working directories for the image_file
        if os.path.exists(special_path):
            shutil.rmtree(special_path)
        os.makedirs(special_path)
        #Copy images to respective working directory
        src_dir = './static/img/'
        for index, row in special_df.iterrows():
            diagnosis = row['type']
            binary_diagnosis = row['binary_type']
            id_code = row['id_code'] + ".png"
            srcfile = os.path.join(src_dir, diagnosis, id_code)
            dstfile = os.path.join(special_path, diagnosis)
            os.makedirs(dstfile, exist_ok = True)
            shutil.copy(srcfile, dstfile)


        Diagnosis = ImageDataGenerator(rescale = 1./255).flow_from_directory(special_path, target_size=(224,224), shuffle = True)

        predicted = retinopathy_model.predict(Diagnosis)
        predicted = np.argmax(predicted,axis=1)

        # Map the label
        Labels = {0: 'Mild', 1: 'Moderate', 2: 'No_DR', 3: 'Proliferate_DR', 4: 'Severe'}
        retinopathy_result = [Labels[k] for k in predicted]
        return render_template('diagnose.html',diabetes_result=diabetes_result,age1 = form_data['Age'], 
            bmi1 = form_data['BMI'],skin1 = form_data['SkinThickness'], ins1 = form_data['Insulin'],
            glu1 = form_data['Glucose'], bp1 = form_data['BloodPressure'], dpf1 = form_data['DiabetesPedigreeFunction'],
            preg1 = form_data['Pregnancies'],retinopathy_result=retinopathy_result[0])
    return render_template('diagnose.html',diabetes_result=diabetes_result,retinopathy_result=retinopathy_result)
    
@app.route('/howitworks')
def howthisworks():
    return render_template('howitworks.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')


if __name__ == "__main__":
    app.run(debug=True)
