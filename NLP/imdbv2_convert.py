"""the imdbv2 ipynb convert for pylint"""

# imports
import re
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from wordcloud import WordCloud

# Download required NLTK data
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")
nltk.download("wordnet")

df = pd.read_csv("movie_data.csv")
df.count()

X = df["review"].values
y = df["sentiment"].values

count = CountVectorizer()

bag_of_words = count.fit_transform(X)

# Get vocabulary (unique words)
feature_names = count.get_feature_names_out()

# Sum the counts for each word across all documents
word_counts = bag_of_words.sum(axis=0).A1

# Create a dictionary mapping words to their counts
word_count_dict = dict(zip(feature_names, word_counts))

print(word_count_dict)


# Define preprocessing functions
def get_wordnet_pos(treebank_tag):
    """function for word of speech tagging. maybe replaced by NLTK pos_tag"""
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    if treebank_tag.startswith("V"):
        return wordnet.VERB
    if treebank_tag.startswith("N"):
        return wordnet.NOUN
    if treebank_tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN


def preprocess_text(text):
    """text preprocessing function"""
    # Tokenize the text
    tokens = word_tokenize(text)

    # POS tagging
    pos_tags = nltk.pos_tag(tokens)

    # Lemmatization and Stemming
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    processed_tokens = []
    for word, tag in pos_tags:
        lemmatized_word = lemmatizer.lemmatize(word, get_wordnet_pos(tag))
        stemmed_word = stemmer.stem(lemmatized_word)
        processed_tokens.append(stemmed_word)

    return " ".join(processed_tokens)


# Apply preprocessing to the dataset
df["processed_review"] = df["review"].apply(preprocess_text)
df["processed_wordCloud"] = df["review"].apply(word_tokenize)


# Split the data into training and test sets
X = df["processed_review"]
y = df["sentiment"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Vectorize the text using TF-IDF
vectorizer = TfidfVectorizer(max_features=10000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# Train a Logistic Regression model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)


# Make predictions on the test set
y_pred = model.predict(X_test_tfidf)


# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {accuracy}")


# Visualize model performance with a confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm, display_labels=["Negative", "Positive"]
)
plt.title("Confusion Matrix")
plt.show()


# Function to predict sentiment of a given text
def predict_sentiment(text, vectorizer_for_predict, model_for_predict):
    """Predict the sentiment from the model"""
    processed_text = preprocess_text(text)
    vectorized_text = vectorizer_for_predict.transform([processed_text])
    pred = model_for_predict.predict(vectorized_text)
    return "Positive" if pred[0] == 1 else "Negative"


# Example sentences
examples = [
    "This movie was fantastic! I really enjoyed it.",
    "I hated this movie. It was terrible.",
    "It was an average movie, not too good, not too bad.",
    "The plot was boring and the acting was bad.",
    "Absolutely loved the cinematography and story.",
]


# Display predictions
for sentence in examples:
    SENTIMENT = predict_sentiment(sentence, vectorizer, model)
    print(f"Review: {sentence}\nPredicted Sentiment: {SENTIMENT}\n")


all_text = re.sub(r"<br\s*/?>", "", " ".join(df["review"]))
positive_text = re.sub(r"<br\s*/?>", "", " ".join(df[df["sentiment"] == 1]["review"]))
negative_text = re.sub(r"<br\s*/?>", "", " ".join(df[df["sentiment"] == 0]["review"]))

wordcloud_all = WordCloud(width=800, height=400, background_color="white").generate(
    all_text
)
wordcloud_positive = WordCloud(
    width=800, height=400, background_color="white"
).generate(positive_text)
wordcloud_negative = WordCloud(
    width=800, height=400, background_color="white"
).generate(negative_text)

# Display the word clouds
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_all, interpolation="bilinear")
plt.title("Word Cloud for All Reviews")
plt.axis("off")
plt.show()


plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_positive, interpolation="bilinear")
plt.title("Word Cloud for Positive Reviews")
plt.axis("off")
plt.show()


plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_negative, interpolation="bilinear")
plt.title("Word Cloud for Negative Reviews")
plt.axis("off")
plt.show()


examples2 = [df["review"][50], df["review"][607], df["review"][101]]

for sentence in examples2:
    SENTIMENT = predict_sentiment(sentence, vectorizer, model)
    print(f"Review: {sentence}\nPredicted Sentiment: {SENTIMENT}\n")


with open("NLP_model.pkl", "wb") as file:
    pickle.dump(model, file)

with open("vectorizer.pkl", "wb") as file:
    pickle.dump(vectorizer, file)

param_grid = {
    "penalty": ["l1", "l2"],
    "C": [0.1, 1, 10],
    "max_iter": [500, 1000, 2000],
    "solver": ["liblinear", "lbfgs", "newton-cg", "sag", "saga"],
}
# X_train_tfidf, y_train
# Perform randomized search over the param_grid
random_search = RandomizedSearchCV(
    model, param_grid, cv=5, scoring="accuracy", n_iter=10, random_state=42
)
random_search.fit(X_train_tfidf, y_train)

# Print the best parameters and the corresponding score
print("Best Parameters: ", random_search.best_params_)
print("Best Score: ", random_search.best_score_)

# Use the best parameters to train a new model on the entire training set
best_log_reg = LogisticRegression(**random_search.best_params_)
best_log_reg.fit(X_train_tfidf, y_train)

# Evaluate the best model on the test set
y_pred = best_log_reg.predict(X_test_tfidf)
print("Test Accuracy: ", accuracy_score(y_test, y_pred))
