import pickle
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the model
model = pickle.load(open("model.pkl", 'rb'))

# Load the vectorizer
vectorizer = pickle.load(open("vectorizer.pkl", 'rb'))

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

# @app.route('/', methods=['GET'])
# def test():
#     if request.method == "GET":
#         return jsonify({"response": "Get request called"})


@app.route('/api/predict', methods=['POST'])
def make_prediction():
    try:
        # Get the JSON data from the POST request
        data = request.get_json()

        # Extract the text data
        text_data = data['data']

        # Transform the text data using the vectorizer
        input_data = vectorizer.transform(text_data)

        # Make prediction using the loaded model
        prediction = model.predict(input_data)

        # Map the numeric predictions to "positive" and "negative"
        prediction_labels = ["positive" if p == 1 else "negative" for p in prediction]

        # Return the prediction as a JSON response
        return jsonify({'prediction': prediction_labels})
    except Exception as e:
        # Return any errors as a JSON response
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0')
