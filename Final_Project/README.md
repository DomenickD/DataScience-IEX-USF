# Data Science Web Application

Welcome to the Data Science Web Application! This app allows users to explore various datasets, utilize machine learning techniques, and make predictions through an interactive web interface.

## Table of Contents
- [Datasets](#datasets)
- [Technologies](#technologies)
- [Machine Learning Techniques](#machine-learning-techniques)
- [Interactive Prediction Tools](#prediction-tools)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Screenshots](#screenshots)

## Datasets
This application utilizes the following datasets:
- **IMDB Dataset**: A collection of 50,000 movie reviews used for sentiment analysis.
- **Titanic Dataset**: Passenger data used to predict survival on the Titanic.
- **Ames Housing Dataset**: Real estate data from Ames, Iowa, used for housing price prediction.
- **MNIST Dataset**: A large database of handwritten digits used for image classification.

## Technologies
The application is built using the following technologies:
- **Streamlit**: The app interface is presented using Streamlit for an interactive user experience.
- **Flask**: Backend routing is managed through Flask, connecting the frontend to the models for predictions and the SQLite database.
- **SQLite Database**: A lightweight database solution used to store and query the datasets mentioned above.
- **Docker Compose**: Manages the multi-container setup, ensuring all services (Streamlit, Flask, SQLite) are orchestrated and run smoothly together in isolated environments.

## Machine Learning Techniques
This app showcases several machine learning techniques, including:
- **Ensemble Techniques**: Methods that combine multiple models to improve accuracy and robustness.
  - *Examples*: Bagging, Boosting, Stacking
- **Unsupervised Learning**: Techniques used to find patterns in data without labeled outcomes.
  - *Examples*: Clustering, Dimensionality Reduction, Anomaly Detection

### TensorFlow vs. PyTorch
- **TensorFlow**: Shorter and easier to implement, ideal for rapid development and deployment.
- **PyTorch**: Offers more customizability, making it a preferred choice for researchers and developers who need finer control over model behavior.

## Interactive Prediction Tools
Explore various prediction tools available in the app:
- **Handwritten Digit Recognition**: A drawable canvas for predicting numbers from the MNIST dataset.
- **Sentiment Analysis**: Analyze movie reviews from the IMDB dataset to predict sentiment.
- **Titanic Survival Prediction**: Predict your chances of survival if you were aboard the Titanic.
- **Housing Price Calculator**: Estimate current housing prices in Ames, Iowa, using historical data.

## Setup and Installation
To run the application locally, follow these steps:

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/DomenickD/DataScience-IEX-USF.git
    ```
    
2. **Navigate to the Project Directory**:
    ```bash
    cd Final_Project
    ```

3. **Run Docker Compose**:
    ```bash
    docker-compose up --build
    ```
    This command will build and start all the necessary containers (Streamlit, Flask, SQLite).
    Ensure your Docker Desktop is running before this command.

4. **Access the Application**:
    Once the containers are running, you can access the application by navigating to `http://localhost:8501` in your web browser.

## Usage
- **Navigate through the App**: Use the sidebar to switch between different datasets and prediction tools.
- **Interact with the Data**: Try out various machine learning techniques and see the predictions in real-time.

## Screenshots
Include screenshots of different sections of the application here:

1. **Home Page**
   ![Home Page 1](Pictures/home_1.png)
   ![Home Page 2](Pictures/home_2.png)

---

2. **Query Tool - Titanic**
   ![titanic_1](Pictures/titanic_1.png)
   ![titanic_2](Pictures/titanic_2.png)
   ![titanic_3](Pictures/titanic_3.png)
   ![titanic_4](Pictures/titanic_4.png)
   ![titanic_5](Pictures/titanic_5.png)
   ![titanic_6](Pictures/titanic_6.png)
   ![titanic_7](Pictures/titanic_7.png)
   ![titanic_8](Pictures/titanic_8.png)
   ![titanic_9](Pictures/titanic_9.png)
   ![titanic_10](Pictures/titanic_10.png)
   ![titanic_11](Pictures/titanic_11.png)
   ![titanic_12](Pictures/titanic_12.png)

---

3. **Query Tool - Housing**
   ![housing_1](Pictures/housing_1.png)
   ![housing_2](Pictures/housing_2.png)
   ![housing_3](Pictures/housing_3.png)
   ![housing_4](Pictures/housing_4.png)
   ![housing_5](Pictures/housing_5.png)
   ![housing_6](Pictures/housing_6.png)
   ![housing_7](Pictures/housing_7.png)
   ![housing_8](Pictures/housing_8.png)
   ![housing_9](Pictures/housing_9.png)

---

4. **Query Tool - Movie Data (IMDB)**
   ![movie_1](Pictures/movie_1.png)
   ![movie_2](Pictures/movie_2.png)
   ![movie_3](Pictures/movie_3.png)
   ![movie_4](Pictures/movie_4.png)

---

5. **Neural Network Comparison**
   ![nn_1](Pictures/nn_1.png)
   ![nn_2](Pictures/nn_2.png)
   ![nn_3](Pictures/nn_3.png)
   ![nn_4](Pictures/nn_4.png)
   ![nn_5](Pictures/nn_5.png)
   ![nn_6](Pictures/nn_6.png)

---

6. **Ensemble Techniques**
   ![ensemble_1](Pictures/ensemble_1.png)
   ![ensemble_2](Pictures/ensemble_2.png)
   ![ensemble_3](Pictures/ensemble_3.png)
   ![ensemble_4](Pictures/ensemble_4.png)

---

7. **Unsupervised Techniques**
   ![unsupervised_1](Pictures/unsupervised_1.png)
   ![unsupervised_2](Pictures/kmeans.gif)
   ![unsupervised_3](Pictures/unsupervised_3.png)

---
8. **Prediction Tool - Titanic**
   ![predict_1](Pictures/predict_1.png)
   ![predict_2](Pictures/predict_2.png)
   ![predict_3](Pictures/predict_3.png)
   ![predict_4](Pictures/predict_4.png)

---

9. **Prediction Tool - Housing**
   ![predict_home_1](Pictures/predict_home_1.png)
   ![predict_home_2](Pictures/predict_home_2.png)
   ![predict_home_3](Pictures/predict_home_3.png)

---

10. **Prediction Tool - Sentiment Analysis**
   ![sentiment_1](Pictures/sentiment_1.png)
   ![sentiment_2](Pictures/sentiment_2.png)

---

11. **Prediction Tool - MNIST**
   ![cnn_1](Pictures/cnn_1.png)
   ![cnn_2](Pictures/cnn_2.png)
   ![cnn_3](Pictures/cnn_3.png)


---