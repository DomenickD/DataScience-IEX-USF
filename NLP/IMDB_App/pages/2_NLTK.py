import streamlit as st

st.header("NLTK to the Rescue!")
st.write("""
NLTK is like a professional organizer for your text:

* **Built-in Tools:**  It provides ready-to-use functions for each step.
* **Reliable:**  The functions are tested and optimized.
* **Consistent:**  Everyone gets the same results using NLTK.
""")
st.divider()

# Section 4: Showcasing NLTK's Advantages (with Examples)
st.header("NLTK in Action: Easy vs. Manual")
st.subheader("Tokenization:")
st.code("words = nltk.word_tokenize('This is a sample sentence.')")
st.write("Vs. writing complex code to split by spaces and punctuation.")

st.subheader("Stopword Removal:")
st.code("words = [w for w in words if not w in stopwords.words('english')]")
st.write("Vs. manually creating and maintaining a list of stopwords.")

st.subheader("Lemmatization:")
st.code("lemmatizer = WordNetLemmatizer()\nwords = [lemmatizer.lemmatize(w) for w in words]")
st.write("Vs. complex algorithms to determine word roots.")