import streamlit as st

st.header("The Natural Language Process")
st.divider()

st.subheader("Text Cleaning")

st.write("""
- Lowercasing: Convert all text to lowercase to ensure consistency and avoid treating words like "Hello" and "hello" as different.

- Punctuation Removal: Remove punctuation marks (periods, commas, exclamation points, etc.) as they often don't contribute much to the meaning of the text in many NLP tasks.

- Number Removal: Decide whether numbers are relevant to your task. If not, remove them.

- Special Character Removal: Get rid of any remaining special characters (e.g., *, &, #) that don't add value.

- Whitespace Handling: Standardize whitespace by converting multiple spaces into single spaces and removing leading or trailing spaces.

- Stopword Removal: Consider removing common words like "the," "a," "an," etc. These words appear frequently but might not be significant for tasks like text classification or topic modeling.
""")
st.write("### Try it yourself!")
input_text = "Try it yourself!"
input_text = st.text_area("Enter your text:", height=150)

if input_text:
    # Lowercasing
    cleaned_text = input_text.lower()
    st.subheader("Lowercased Text:")
    st.code(cleaned_text, language="text")

    # Punctuation Removal
    cleaned_text = "".join(c for c in cleaned_text if c.isalnum() or c.isspace()) 
    st.subheader("Punctuation Removed:")
    st.code(cleaned_text, language="text")

    # Number and Special Character Removal (Combined)
    cleaned_text = "".join(c for c in cleaned_text if c.isalpha() or c.isspace()) 
    st.subheader("Numbers & Special Characters Removed:")
    st.code(cleaned_text, language="text")

# st.code("""
# import re
# def preprocessor(text):
#     text = re.sub('<[^>]*>', '', text)
#     emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',
#                            text)
#     text = (re.sub('[\W]+', ' ', text.lower()) +
#             ' '.join(emoticons).replace('-', ''))
#     return text
# """)
# st.caption("This is one such function the helps to remove emoticons from reviews. We only want to deal with words.")

st.divider()
st.subheader("Tokenization")
st.write("""
**A "Token" is broadly defined as 3-4 characters.**         

- Word Tokenization: Break the text into individual words. You can do this manually by splitting the text at whitespace boundaries or using custom logic for handling punctuation.

- Sentence Tokenization: If needed, divide the text into separate sentences. Look for punctuation marks like periods, question marks, and exclamation points as potential sentence boundaries.
""")

st.write("### Try it yourself!")

if input_text:
    # Word Tokenization
    word_tokens = cleaned_text.split()
    st.subheader("Word Tokens:")
    st.write(word_tokens)

    # Sentence Tokenization (Simplified)
    sentences = cleaned_text.split(".")  # Basic split on periods
    st.subheader("Sentence Tokens (Basic):")
    st.write(sentences)

st.divider()
st.subheader("Normalization")

st.write("""

- Stemming: Reduce words to their root form (e.g., "running," "runs," "ran" become "run"). This can help group similar words together.

- Lemmatization: A more sophisticated approach than stemming, lemmatization reduces words to their base or dictionary form (lemma) considering the part of speech. For example, "better" becomes "good."
""")

st.divider()

