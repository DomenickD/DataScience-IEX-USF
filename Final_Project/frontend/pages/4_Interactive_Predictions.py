import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import numpy as np
import pandas as pd
import requests

st.header("Prediction Tool!")

table_option = st.radio(
    "Select what dataset you would like to obtain a prediction from:",
    ("Titanic", "Housing", "Movie", "MNIST"),
)

st.divider()

if table_option == "Titanic":
    st.header("How would you fare if you were on the Titanic?")

    st.subheader("Survival Predictor Tool")

    sex_map = {"male": 0, "female": 1}

    pclass = st.selectbox("Passenger Class:", [1, 2, 3])
    sex = st.selectbox("Sex:", ["male", "female"])
    age = st.slider("Age:", 0, 100, 30)
    Fare = st.slider("Fare:", 0, 512, 100)
    sib = st.selectbox("Number of Siblings/Spouses:", ["0", "1", "2", "3"])
    par = st.selectbox("Number of Parents/Children:", ["0", "1", "2", "3"])

    user_input = pd.DataFrame(
        {
            "Pclass": [pclass],
            "Age": [age],
            "SibSp": [sib],
            "Parch": [par],
            "Fare": [Fare],
            "Sex_binary": sex_map[sex],
        }
    ).to_dict(orient="records")[0]

    if st.button("Predict"):
        response = requests.post(
            "http://flask_route:5000/predict_titanic", json=user_input
        )
        result = response.json()
        result_df = pd.DataFrame([result])  # Convert the response to a DataFrame
        # st.write(result_df)
        survival_prob = result_df["survival_prob"][0]
        prediction = result_df["survived"][0]

        # Display the outcome
        if prediction == 1:
            st.success("You Survived!")
            st.balloons()
            st.image("Pictures/I_surived.png")
        else:
            st.error("You did not survive...")
            st.snow()
            st.image("Pictures/wasted.png")
            st.caption("Cold out here... huh?")

elif table_option == "Housing":
    st.write(
        """
    ## Housing Price Exploration Tool

    ---
            
    Let's explore factors that might influence housing prices. 
    """
    )

    lot_area = st.slider("Lot Area (sq. ft.):", 0, 20000, 5000)
    overall_qual = st.select_slider(
        "Overall Quality (1-10):", options=range(1, 11), value=5
    )
    total_bsmt_sf = st.slider("Total Basement SF (sq. ft.):", 0, 3000, 1000)
    gr_liv_area = st.slider("Above Grade Living Area (sq. ft):", 0, 5000, 1500)
    full_bath = st.number_input("Full Bathrooms:", min_value=0, max_value=5, value=2)
    TotRms_AbvGrd = st.number_input(
        "Total rooms above grade:", min_value=0, max_value=5, value=2
    )
    Fireplaces = st.number_input(
        "Number of fireplaces:", min_value=0, max_value=5, value=2
    )

    # columns = ['Gr Liv Area', 'Total Bsmt SF', 'Full Bath', 'TotRms AbvGrd', 'Fireplaces', 'Lot Area', 'Overall Qual', 'SalePrice']

    user_input = pd.DataFrame(
        {
            "Lot Area": [lot_area],
            "Overall Qual": [overall_qual],
            "Total Bsmt SF": [total_bsmt_sf],
            "Gr Liv Area": [gr_liv_area],
            "Full Bath": [full_bath],
            "TotRms AbvGrd": [TotRms_AbvGrd],
            "Fireplaces": [Fireplaces],
        }
    ).to_dict(orient="records")[0]

    if st.button("Predict"):
        response = requests.post(
            "http://flask_route:5000/predict_housing", json=user_input
        )
        result = response.json()
        result_df = pd.DataFrame([result])
        prediction = result_df["price"][0]

        st.subheader("Predicted Sale Price")
        st.write(f"${prediction:,.2f}")

elif table_option == "Movie":
    query = "This is a good sentiment!"
    query = st.text_area(label="Enter ONE sentence for sentiment testing:", value=query)
    if st.button("Predict"):
        response = requests.post(
            "http://flask_route:5000/predict_sentiment", json={"text": query}
        )
        result = response.json()
        sentiment = result["sentiment"]
        if sentiment == "Positive":
            st.success("The sentiment is Positive!")
        elif sentiment == "Negative":
            st.error("The sentiment is Negative!")


elif table_option == "MNIST":
    st.title("Handwritten Digit Recognition")

    # Create a canvas component
    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width=3,
        stroke_color="#FFFFFF",
        background_color="#000000",
        update_streamlit=True,
        height=150,
        width=150,
        drawing_mode="freedraw",
        key="canvas",
    )

    if st.button("Predict"):
        if canvas_result.image_data is not None:
            # Convert the image data to a PIL Image
            input_numpy_array = np.array(canvas_result.image_data)
            input_image = Image.fromarray(input_numpy_array.astype("uint8"), "RGBA")
            input_image.save("user_drawing.png")
            # Convert to grayscale
            gray_image = input_image.convert("L")
            # Resize the image to 28x28
            processed_img = gray_image.resize((28, 28))
            # Send to Flask API
            image_data = np.array(processed_img).tolist()
            response = requests.post(
                "http://flask_route:5000/predict_digit", json={"image_data": image_data}
            )
            result = response.json()
            predicted_digit = result["digit"]
            st.write(f"Predicted Digit: {predicted_digit}")
