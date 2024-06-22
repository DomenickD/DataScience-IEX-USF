# import streamlit as st
# import nltk
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
# from nltk import pos_tag
# from openai import OpenAI
# import pandas as pd

# # Initialize the OpenAI client pointing to the local LM Studio server
# client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

# # Title and introduction
# st.title("NLTK vs AI Preprocessing Comparison")
# st.write("Enter a sentence to compare preprocessing steps using NLTK and AI supported by LM Studio.")

# # Function to preprocess text using NLTK
# def nltk_preprocess(text):
#     tokens = word_tokenize(text)
#     stop_words = set(stopwords.words('english'))
#     filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
#     pos_tags = pos_tag(filtered_tokens)
#     return {
#         "tokens": tokens,
#         "filtered_tokens": filtered_tokens,
#         "pos_tags": pos_tags
#     }

# # Function to preprocess text using AI supported by LM Studio
# def ai_preprocess(text):
#     response = client.chat.completions.create(
#         model="lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF",
#         messages=[
#             {"role": "system", "content": "You are a code assistant."},
#             {"role": "user", "content": f"Preprocess the following text and provide tokens, filtered tokens without stopwords, and part of speech tags: '{text}'"}
#         ],
#         temperature=0.7,
#     )
#     result = response.choices[0].message.content
#     return eval(result)  # Assuming the AI returns a Python dictionary as a string

# # Accept user input
# sentence = st.text_input("Enter a sentence:")

# if sentence:
#     # Preprocess the sentence using NLTK
#     nltk_results = nltk_preprocess(sentence)

#     # Preprocess the sentence using AI supported by LM Studio
#     try:
#         ai_results = ai_preprocess(sentence)
#     except Exception as e:
#         st.error(f"An error occurred: {e}") #I want to use this if we can catch any errors. for presentation, I want to use a set string.
#         ai_results = {"tokens": [], "filtered_tokens": [], "pos_tags": []}

#     # Display the results side by side in a table
#     # df = pd.DataFrame({
#     #     "NLTK": ["Tokens", "Filtered Tokens", "POS Tags"],
#     #     "NLTK Results": [nltk_results["tokens"], nltk_results["filtered_tokens"], nltk_results["pos_tags"]],
#     #     "AI": ["Tokens", "Filtered Tokens", "POS Tags"],
#     #     "AI Results": [ai_results.get("tokens", []), ai_results.get("filtered_tokens", []), ai_results.get("pos_tags", [])]
#     # })

#         # Convert lists to strings for display in the DataFrame
#     def stringify(items):
#         return ', '.join(str(item) for item in items)
    
#     df = pd.DataFrame({
#         "Step": ["Tokens", "Filtered Tokens", "POS Tags"],
#         "NLTK Results": [stringify(nltk_results["tokens"]), stringify(nltk_results["filtered_tokens"]), stringify(nltk_results["pos_tags"])],
#         "AI Results": [stringify(ai_results.get("tokens", [])), stringify(ai_results.get("filtered_tokens", [])), stringify(ai_results.get("pos_tags", []))]
#     })


#     st.table(df)

import streamlit as st
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
import re

st.header("NLTK (Manual) vs. LLM (LLama) Performance")
nltk.download('punkt')
nltk.download('stopwords')
st.divider()
st.subheader("Preprocessing")
#The sentence for processing. If we change this, we need to ask the LLM so preprocess that one too
sentence = "This is an example of how NLP works~! Can it clean numbers? 1 2 3 %^&* :)"
#remove re
cleaned_text = re.sub(r'[^a-zA-Z\s]+', '', sentence) 
#lower
lower_sentence = cleaned_text.lower()
#tokens
tokens = word_tokenize(lower_sentence)
#stopword remove
stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if not word in stop_words]
# pos tagging
pos_word = nltk.pos_tag(filtered_tokens)
#stemming
stemmer = PorterStemmer()
stemmed_words = [stemmer.stem(word) for word in filtered_tokens]
# lemm
lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_tokens]


st.write(f"""
The text for processing: {sentence}

| NLP Step | NLTK | LLama (LLM) |
|---|---|---|
| Lower Case | {lower_sentence} *(Also manually removed non-text characters)* | this is an example of how nlp works~! can it clean numbers? 1 2 3 %^&* :) |
| Tokenization | {tokens} | [this, is, an, example, of, how, nlp, works~, can, it, clean, numbers?, 1, 2, 3, %, ^%, &, *, :) ] |
| Stopwords | {filtered_tokens} | [example, how, works, it, clean] (removed: is, an, of, nlp, numbers, this, can) |
| POS Tagging | {pos_word} | [this (DT), is (VBZ), an (DT), example (NN), of (IN), how (WRB), nlp (NNP), works~! (.!), can (MD), it (PRP), clean (VB), numbers? (?) 1 (CD) 2 (CD) 3 (CD) % (%), ^ (X), & (&), * (*), :) (:)] |
| Stemming | {stemmed_words} | [this, is, an, exampl, how, nlp, work, can, it, clea, numb] (removed: example, of, nlp, works~!, clean, numbers?) |
| Lemitzation | {lemmatized_words} | [this, be, a, way, nlp, do, something, can, you, make, clean, number] (removed: is, an, exampl, how, work, it, clea, numb) |
""")
st.divider()
st.subheader("Sentiment Prediction")

review_one = "How this character ever got hired by the post office is far beyond me. The test that postal workers take is so difficult. There is no way that a guy this stupid can work at the post office. Everyone in this movie is just stupid and that is probably the point of the movie. How they could go their entire lives and not see an elevator is also puzzling. I didn't take this movie too seriously but it was so stupid. Then he tries to start the car without his keys? Lots of horrible scenes and horrible acting and this movie is not funny at all. It's just a sad stupid mess. I liked the moms dress though.<br /><br />Send it back to sender as soon as possible."

review_two = "When this cartoon first aired I was under the impression that it would be at least half way descent, boy was I wrong. I must admit watching this cartoon is almost as painful as watching Batman and Robin with George Clooney all those years ago. I watched a few episodes and two of them had Batman literally get his ass kicked left and right by the Penguin who fought like Jet Li and beat the crap out of Batman and I watched another episode where Batman got his butt kicked again by the Joker, who apparently was using Jackie Chan moves while flipping in the air like a ninja. Since when were the Joker or the Penguin ever a match for Batman ? and worse yet when were Joker and Penguin Kung Fu counterparts of Jackie Chan and Jet Li. It's truly embarrassing, depressing and sad the way the image of Batman is portrayed in this show. The animation is awful and the dialog is terrible. Being a Batman fan since my boyhood I can honestly and strongly advise you to stay away and avoid this show at all cost, because it doesn't project the true image of Batman. This cartoon is more like a wannabe Kung Fu Flick and if you really wanna see a classic Batman cartoon I strongly recommend Batman the Animated Series, but this cartoon is nothing more than a piece of S---T! Get Batman: The Animates Series and don't waste your time with this cartoon."

review_three = 'Good, funny, straightforward story, excellent Nicole Kidman (I almost always like the movies she\'s in). This was a good "vehicle" for someone adept at comedy and drama since there are elements of both. A romantic comedy wrapped around two crime stories, great closing lines. Chaplin, very good here, was also good in another good, but unpopular romantic comedy ("Truth about Cats & Dogs"). Maybe they\'re too implausible. Ebert didn\'t even post a review for this. The great "screwball" comedies obviously were totally implausible ("Bringing up Baby", etc.). If you\'ve seen one implausible comedy, you\'ve seen them all? Or maybe people are ready to move on from the 1930s. Weird. Birthday Girl is a movie I\'ve enjoyed several times. Nicole Kidman may be the "killer app" for home video.'

st.write(f"""
         
| Review Original | Model Prediction | LLama (LLM) Prediction |
|---|---|---|
| {review_one} | Negative | "overwhelmingly NEGATIVE" |
| {review_two} | Negative | "overwhelmingly NEGATIVE" |
| {review_three} | Positive | "POSITIVE" |

         
""")
st.caption("What do the predictions mean?")

st.divider()

st.write("The sentiment analysis is a simple use case for NLP so we see that the LLama LLM performed the same as our trained model when it came down to a binary classifcation (prediction).")