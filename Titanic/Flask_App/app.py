from flask import Flask, render_template, request, redirect, url_for, session, flash
import pandas as pd
import pickle

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Load model and scaler from files
model = pickle.load(open('my_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    else:
        return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        if request.form['password'] == 'IEX123' and request.form['username'] == 'User1':
            session['logged_in'] = True
            return redirect(url_for('home'))
        else:
            flash('Wrong username or password!')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session['logged_in'] = False
    return redirect(url_for('home'))

@app.route('/predict', methods=['POST'])
def predict():
    # Prediction logic goes here
    pass

if __name__ == '__main__':
    app.run(debug=True)
