import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import requests

# Load the dataset
df = pd.read_csv("NLP/movie_data.csv")

# Take a random sample of 500 reviews
df_sample = df.sample(n=2000, random_state=42)

# Split the sample dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df_sample["review"], df_sample["sentiment"], test_size=0.2, random_state=42
)

# Vectorize the text data
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

# Train a logistic regression model
logreg = LogisticRegression()
logreg.fit(X_train_vect, y_train)

# Predict and evaluate the logistic regression model
y_pred_logreg = logreg.predict(X_test_vect)
accuracy_logreg = accuracy_score(y_test, y_pred_logreg)
print(f"Logistic Regression Accuracy: {accuracy_logreg}")


# Function to get predictions from the LLM
def get_llm_predictions(reviews):
    predictions = []
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    url = f"{base_url}/completions"
    for review in reviews:
        payload = {
            "model": "model-identifier",
            "prompt": f"Classify the sentiment of this review as positive or negative: {review}\n",
            "max_tokens": 10,
            "temperature": 0.0,
        }
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code != 200:
            print(f"Error: {response.status_code} - {response.text}")
            predictions.append(-1)
            continue
        completion = response.json()
        if "choices" not in completion:
            print(f"Unexpected response format: {completion}")
            predictions.append(-1)
            continue
        prediction = completion["choices"][0]["text"].strip().lower()
        if "positive" in prediction:
            predictions.append(1)
        elif "negative" in prediction:
            predictions.append(0)
        else:
            predictions.append(-1)  # For unknown responses
    return predictions


# Get predictions from the LLM for the test set
api_key = "lm-studio"
base_url = "http://localhost:1234/v1"
X_test_list = X_test.tolist()
y_pred_llm = get_llm_predictions(X_test_list)

# Evaluate the LLM predictions
valid_indices = [i for i, pred in enumerate(y_pred_llm) if pred != -1]
y_test_filtered = y_test.iloc[valid_indices]
y_pred_llm_filtered = [y_pred_llm[i] for i in valid_indices]

accuracy_llm = accuracy_score(y_test_filtered, y_pred_llm_filtered)
print(f"LLM Accuracy: {accuracy_llm}")
