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
    error = None
    if request.method == 'POST':
        if request.form['username'] == 'User1' and request.form['password'] == 'IEX123':
            session['logged_in'] = True
            flash('You were just logged in!')
            return redirect(url_for('home'))
        else:
            error = 'Invalid credentials. Please try again.'
    return render_template('login.html', error=error)

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    flash('You were just logged out!')
    return redirect(url_for('login'))

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Extract form data
        pclass = request.form['pclass']
        sex = request.form['sex']
        age = request.form['age']
        # sibsp = request.form['sibsp']
        # parch = request.form['parch']
        fare = request.form['fare']

        sex_map = {'male': 0, 'female': 1}
        sex = sex_map[sex]

        # Assume preprocessing and model prediction logic here
        # For example:
        # features = np.array([[pclass, sex, age, sibsp, parch, fare]])
        # scaled_features = scaler.transform(features)
        # prediction = model.predict(scaled_features)

        prediction = "Survived"  # Placeholder

        # Render the same prediction page with the result
        return render_template('predict.html', prediction=prediction)

    return redirect(url_for('predict'))


if __name__ == '__main__':
    app.run(debug=True)
