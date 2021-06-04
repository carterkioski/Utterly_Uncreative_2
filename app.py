#import dependencies
from flask import Flask, render_template, redirect, request

import os

#creating an app
app = Flask(__name__)


#Homepage
@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/diagnose', methods = ['POST', 'GET'])
def diagnose():
    if request.method == 'POST':
        form_data = request.form
        #here take form data , clean it up, pass it through the two models
        #result = responses from models
        return render_template('diagnose.html', result=True)
    return render_template('diagnose.html')

@app.route('/howitworks')
def howthisworks():
    return render_template('howitworks.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')


if __name__ == "__main__":
    app.run(debug=True)
