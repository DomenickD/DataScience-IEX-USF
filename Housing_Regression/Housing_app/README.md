# Ames Housing Regression Analysis

## **Introduction**

This Streamlit application empowers you to explore the Ames Housing Dataset, a comprehensive collection of residential property sales in Ames, Iowa. Using a trained machine learning model, you can interactively predict housing prices based on various features like living area, number of bedrooms, and overall quality. Additionally, the app provides visualizations to help you uncover relationships between different housing attributes and their impact on sale price.

## Running the Project

**1. Requirements**

* Python 3.12
* Streamlit (`pip install streamlit`)
* Pandas (`pip install pandas`)
* Pickle (`pip install pickle`) - for loading the pre-trained model
* Plotly (`pip install plotly`) - for interactive charts
* Numpy (`pip install numpy`)
* Matplotlib (`pip install matplotlib`)
* Seaborn (`pip install seaborn`) - Heatmaps
* Scikit-learn (`pip install scikit-learn`)
* xgboost (`pip install xgboost`)
* statsmodels  (`pip install statsmodels`)

**2. Running with Docker**

(Assuming you have Docker installed)

* Build the Docker image:

```docker build -t housing-app .```

* Run the container:

```docker run -p 8501:8501 housing-app```

Open http://localhost:8501 in your web browser to access the Streamlit app.

**3. Running Locally**

* Clone this repository.

* Install the required libraries (pip install -r requirements.txt).

* Run the app using:

```streamlit run Housing_app.py```

## Conclusions

* The Features that had the largest impact of teh Sale Price are: Above-grade living area (sq. ft.), Rates overall material and finish and Total basement area (sq. ft.).

* Surprisingly, the size of the lot did not play as large a role in the Sale Price as previously thought.

* The majority of homes were prices between $100k and $300k. Numbers became fewer the higher the price. 

* Overall Quality was the feature that was surprising. Higher quality did not always mean higher prices. People often paid more for less. 
