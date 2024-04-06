# Titanic Passenger Survival Analysis

## **Introduction**

This Streamlit application is an interactive exploration of the Titanic passenger survival dataset. It allows users to input passenger characteristics and predict their likelihood of survival based on a trained machine learning model. Additionally, the app visualizes survival patterns across different passenger classes.

## Running the Project

**1. Requirements**

* Python 3.10
* Streamlit (`pip install streamlit`)
* Pandas (`pip install pandas`)
* Pickle (`pip install pickle`) - for loading the pre-trained model
* Plotly (`pip install plotly`) - for interactive charts
* Numpy (`pip install numpy`)
* Matplotlib (`pip install matplotlib`)
* Seaborn (`pip install seaborn`) - Heatmaps
* Scikit-learn (`pip install scikit-learn`)

**2. Running with Docker**

(Assuming you have Docker installed)

* Build the Docker image:

```docker build -t titanic_survival_analysis .```

* Run the container:

```docker run -p 8501:8501 titanic_survival_analysis```

Open http://localhost:8501 in your web browser to access the Streamlit app.

**3. Running Locally**

* Clone this repository.

* Install the required libraries (pip install -r requirements.txt).

* Run the app using:

```streamlit run Titanic_app.py```

## Conclusions

* First Class Passengers had a higher chance of survival.

* Women and children had a higher chance of survival.

* Survival chance decreased with age.

* The higher the fare that was paid, the better chance of survival. 
