import streamlit as st

st.title("Welcome to the Data Science App")
st.write("By: Domenick Dobbs")

st.divider()

st.markdown(
    """
### Datasets
- **IMDB Dataset**: A collection of 50,000 movie reviews for sentiment analysis.
- **Titanic Dataset**: Passenger data used to predict survival on the Titanic.
- **Ames Housing Dataset**: Real estate data from Ames, Iowa, used for housing price prediction.
- **MNIST Dataset**: A large database of handwritten digits for image classification.

### Technologies
- **Streamlit**: The app interface is presented using Streamlit for an interactive user experience.
- **Flask**: Backend routing is managed through Flask, connecting the frontend to the models for predictions and the sqlite database.
- **SQLite Database**: A lightweight database solution used to store and query the datasets mentioned above.
- **Docker Compose**: Manages the multi-container setup, ensuring all services (Streamlit, Flask, SQLite) are orchestrated and run smoothly together in isolated environments.

### Techniques
- **Neural Network Framework Comparison**: Side-by-side comparisons of models built with PyTorch and TensorFlow.
- **Unsupervised Learning**: Techniques applied to explore and cluster the data.
- **Ensemble Learning**: Both regression and classification tasks are tackled using ensemble methods.
- **Data Visualizations**: Various visualizations provide insights into the datasets.

### Prediction Tool Page
Explore the rudimentary suite of predictive tools:
- **Titanic Survival Prediction**: Find out if you would have survived the Titanic disaster.
- **Housing Price Calculator**: Estimate current housing prices in Ames, Iowa, using historical data.
- **Sentiment Analysis**: Analyze movie reviews from the IMDB dataset to predict sentiment.
- **Handwritten Digit Recognition**: Use a drawable canvas to predict numbers from the MNIST dataset.

---

"""
)
