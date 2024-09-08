"""
This file is responsible for the heavy lifting, flask routing,
and model predictions from loaded models. All logic here. 
"""

import sqlite3
import pickle
from flask import Flask, request, jsonify
import keras
import numpy as np
import pandas as pd
from flask_cors import CORS
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer

# Establish the app instance
app = Flask(__name__)
# CORS stands for Cross-Origin Resource Sharing.
# It's a security mechanism implemented in web browsers that restricts
# web pages from making requests to a different domain than the one they originated from.
CORS(app)

# Load models and vectorizer
with open("models/vectorizer.pkl", "rb") as vec_file:
    vectorizer = pickle.load(vec_file)

with open("models/NLP_model.pkl", "rb") as model_file:
    nlp_model = pickle.load(model_file)

FILENAME = "models/mnist_model.keras"
with open(FILENAME, "rb") as f:
    mnist_model = keras.models.load_model(FILENAME)

with open("models/my_model.pkl", "rb") as f:
    titanic_model = pickle.load(f)

with open("models/scaler.pkl", "rb") as f:
    titanic_scaler = pickle.load(f)

with open("models/xgb_pipeline_minmaxscaler.pkl", "rb") as f:
    housing_pipeline = pickle.load(f)

# mnist_model = keras.models.load_model("models/mnist_model.keras")
# titanic_model = pickle.load(open("models/my_model.pkl", "rb"))
# titanic_scaler = pickle.load(open("models/scaler.pkl", "rb"))
# housing_pipeline = pickle.load(open("models/xgb_pipeline_minmaxscaler.pkl", "rb"))

# NLTK downloads
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("averaged_perceptron_tagger")
nltk.download("averaged_perceptron_tagger_eng")
nltk.download("wordnet")


def preprocess_text(text):
    """
    Perform NLP preprocessing on the given text, including tokenization,
    lemmatization, and stemming. Defaults to lemmatizing as nouns.
    """
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()

    tokens = word_tokenize(text)

    processed_tokens = [stemmer.stem(lemmatizer.lemmatize(word)) for word in tokens]

    return " ".join(processed_tokens)


# The SQLITE DB query
def query_database(query_intake):
    """
    This is the function to do the actual database querying.
    """
    conn = sqlite3.connect("/app/RA_projects.db")  # Adjust path if needed
    cursor = conn.cursor()
    cursor.execute(query_intake)
    columns = [description[0] for description in cursor.description]  # Get column names
    data = cursor.fetchall()
    conn.close()
    return {"columns": columns, "data": data}


# More functions but these will recieve the POST requests and do the `magic`
@app.route("/predict_sentiment", methods=["POST"])
def predict_sentiment():
    """
    Function to predict the sentiment and return a json response.
    """
    data = request.json
    text = data.get("text", "")
    processed_text = preprocess_text(text)
    vectorized_text = vectorizer.transform([processed_text])
    pred = nlp_model.predict(vectorized_text)
    sentiment = "Positive" if pred[0] == 1 else "Negative"
    return jsonify({"sentiment": sentiment})


@app.route("/predict_digit", methods=["POST"])
def predict_digit():
    """
    This function handles the canvas input for predicting a numbered single digit using MNIST data.
    """
    data = request.json
    image_data = np.array(data.get("image_data"))
    processed_img = np.array(image_data).reshape((1, 28, 28, 1)) / 255.0
    prediction = mnist_model.predict(processed_img)
    predicted_digit = np.argmax(prediction)
    return jsonify({"digit": int(predicted_digit)})


@app.route("/predict_titanic", methods=["POST"])
def predict_titanic():
    """
    This function takes in an array of data and uses it to
    predict against a pretrained model for the titanic data.
    """
    data = request.json
    # df = pd.DataFrame(data, index=[0])
    df = pd.DataFrame([data])
    scaled_data = titanic_scaler.transform(df)
    prediction = titanic_model.predict(scaled_data)[0]
    survival_prob = titanic_model.predict_proba(scaled_data)[0][1]
    return jsonify({"survived": int(prediction), "survival_prob": float(survival_prob)})


@app.route("/predict_housing", methods=["POST"])
def predict_housing():
    """
    This function takes in an array of data and uses it to
    predict against a pretrained model for the Housing data.
    """
    data = request.json
    # df = pd.DataFrame(data, index=[0])
    df = pd.DataFrame([data])
    prediction = housing_pipeline.predict(df)[0]
    return jsonify({"price": float(prediction)})


@app.route("/query", methods=["POST"])
def query_route():
    """
    This function support sthe handling of
    SQL queries for the database file in another docker container.
    """
    try:
        query = request.json.get("query")
        if not query:
            return jsonify({"error": "No query provided"}), 400
        data = query_database(query)
        return jsonify(data)
    except sqlite3.DatabaseError as db_error:
        return jsonify({"error": f"Database error: {str(db_error)}"}), 500
    except ValueError as val_error:
        return jsonify({"error": f"Value error: {str(val_error)}"}), 400
    except KeyError as e:
        return jsonify({"error": f"KeyError: {str(e)}"}), 400  # Bad Request
    except IOError as e:
        return jsonify({"error": f"IOError: {str(e)}"}), 500  # Internal Server Error


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
