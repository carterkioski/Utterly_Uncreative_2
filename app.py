#import dependencies
from flask import Flask, render_template, redirect

import os

#creating an app
app = Flask(__name__)


#Homepage
@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/diagnose')
def diagnose():
    return render_template('diagnose.html')

@app.route('/how-this-works')
def howthisworks():
    return render_template('howthisworks.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')


if __name__ == "__main__":
    app.run(debug=True)