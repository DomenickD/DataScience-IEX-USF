import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler

# min_max_scaler = MinMaxScaler()
st.header("How would you fare if you were on the Titanic?")

st.subheader("Survival Predictor Tool")

sex_map = {'male': 0, 'female': 1}

pclass = st.selectbox('Passenger Class:', [1, 2, 3])
sex = st.selectbox('Sex:', ['male', 'female'])
age = st.slider('Age:', 0, 100, 30)
Fare = st.slider('Fare:', 0, 512, 100)

user_input = pd.DataFrame({
    'Age': [age],
    'Sex_binary': sex_map[sex],
    'Pclass': [pclass],
    'Fare': [Fare]
})

with open('my_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    min_max_scaler = pickle.load(f)

if st.button("Predict"):
    user_input_scaled = min_max_scaler.transform(user_input)

    prediction = model.predict(user_input_scaled)[0]
    survival_prob = model.predict_proba(user_input_scaled)[0][1] 
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

