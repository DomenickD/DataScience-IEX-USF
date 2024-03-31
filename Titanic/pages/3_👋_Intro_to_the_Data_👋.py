import streamlit as st
import pandas as pd
import pickle

st.header("How would you fair if you were on the Titanic?")

st.subheader("Survival Predictor Tool")

sex_map = {'male': 0, 'female': 1}

pclass = st.selectbox('Passenger Class:', [1, 2, 3])
sex = st.selectbox('Sex:', ['male', 'female'])
age = st.slider('Age:', 0, 100, 30)
Fare = st.slider('Fare:', 0, 512, 100)

user_input = pd.DataFrame({
    'Pclass': [pclass],
    'Sex': sex_map[sex],
    'Age': [age],
    'Fare': [Fare]
})

with open('my_model.pkl', 'rb') as f:
    model = pickle.load(f)

# prediction = model.predict(user_input)[0]

if st.button("Predict"):
    prediction = model.predict(user_input)[0]
    survival_prob = model.predict_proba(user_input)[0][1] 
    not_survived_prob = 1 - survival_prob

    # Display the outcome
    if prediction == 1:
        st.success("You Survived!")
        st.balloons()
    else:
        st.error("You did not survive...")
        st.markdown("""
                    <h2 style='text-align: center; color: red;'> ❌ You did not survive... </h2> 
                    """, unsafe_allow_html=True)

    # Display the probabilities
    # st.write("Probabilities:")
    # st.write(f"Survival: {survival_prob:.2f}")  # Format to 2 decimal places 
    # st.write(f"Not Survived: {not_survived_prob:.2f}") 


