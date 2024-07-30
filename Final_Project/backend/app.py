#Import everything needed in one place
from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import tensorflow as tf
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer, PorterStemmer
import nltk
import sqlite3
import pandas as pd
import xgboost as xgb

#Establish the app instance
app = Flask(__name__)
#CORS stands for Cross-Origin Resource Sharing. It's a security mechanism implemented in web browsers that restricts web pages from making requests to a different domain than the one they originated from.
CORS(app)

# Load models and vectorizer
with open('models/vectorizer.pkl', 'rb') as vec_file:
    vectorizer = pickle.load(vec_file)

with open('models/nlp_model.pkl', 'rb') as model_file:
    nlp_model = pickle.load(model_file)

mnist_model = tf.keras.models.load_model('models/mnist_model.keras')
titanic_model = pickle.load(open('models/my_model.pkl', 'rb'))
titanic_scaler = pickle.load(open('models/scaler.pkl', 'rb'))
housing_pipeline = pickle.load(open('models/xgb_pipeline_minmaxscaler.pkl', 'rb'))

# NLTK downloads
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

#List out all the functions I am going to use!
#NLP Pre-Preprocessing
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
#The actual NLP Preprocessing
def preprocess_text(text):
    tokens = word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    processed_tokens = []
    for word, tag in pos_tags:
        lemmatized_word = lemmatizer.lemmatize(word, get_wordnet_pos(tag))
        stemmed_word = stemmer.stem(lemmatized_word)
        processed_tokens.append(stemmed_word)
    return ' '.join(processed_tokens)

#The SQLITE DB query 
def query_database(query):
    conn = sqlite3.connect('/app/RA_projects.db')  # Adjust path if needed
    cursor = conn.cursor()
    cursor.execute(query)
    columns = [description[0] for description in cursor.description]  # Get column names
    data = cursor.fetchall()
    conn.close()
    return {"columns": columns, "data": data}

#More functions but these will recieve the POST requests and do the `magic`
@app.route('/predict_sentiment', methods=['POST'])
def predict_sentiment():
    data = request.json
    text = data.get('text', '')
    processed_text = preprocess_text(text)
    vectorized_text = vectorizer.transform([processed_text])
    pred = nlp_model.predict(vectorized_text)
    sentiment = 'Positive' if pred[0] == 1 else 'Negative'
    return jsonify({'sentiment': sentiment})

@app.route('/predict_digit', methods=['POST'])
def predict_digit():
    data = request.json
    image_data = np.array(data.get('image_data'))
    processed_img = np.array(image_data).reshape(1, 28, 28, 1) / 255.0
    prediction = mnist_model.predict(processed_img)
    predicted_digit = np.argmax(prediction)
    return jsonify({'digit': int(predicted_digit)})

@app.route('/predict_titanic', methods=['POST'])
def predict_titanic():
    data = request.json
    # df = pd.DataFrame(data, index=[0])
    df = pd.DataFrame([data])
    scaled_data = titanic_scaler.transform(df)
    prediction = titanic_model.predict(scaled_data)[0]
    survival_prob = titanic_model.predict_proba(scaled_data)[0][1]
    return jsonify({'survived': int(prediction), 'survival_prob': float(survival_prob)})

@app.route('/predict_housing', methods=['POST'])
def predict_housing():
    data = request.json
    # df = pd.DataFrame(data, index=[0])
    df = pd.DataFrame([data])
    prediction = housing_pipeline.predict(df)[0]
    return jsonify({'price': float(prediction)})


@app.route('/query', methods=['POST'])
def query():
    try:
        query = request.json.get('query')
        if not query:
            return jsonify({"error": "No query provided"}), 400
        data = query_database(query)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
