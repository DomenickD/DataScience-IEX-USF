import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler

from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import train_test_split as tts
import xgboost as xgb #pip install xgboost
from sklearn import metrics


columns = ['Gr Liv Area', 'Total Bsmt SF', 'Full Bath', 'TotRms AbvGrd', 'Fireplaces', 'Lot Area', 'Overall Qual', 'SalePrice']

df = pd.read_csv('AmesHousing.txt',
                 sep='\t',
                 usecols=columns)

# df["Central_Air_Binary"] = df['Central Air'].map({'N': 0, 'Y': 1})
# df = df.drop("Central Air", axis = 1)

df = df.dropna(axis=0)
Y = df.SalePrice
X = df.drop("SalePrice", axis = 1)
Scalerminmax = MinMaxScaler()
df_scaled = Scalerminmax.fit_transform(X)

X_train, X_test, y_train, y_test = tts(df_scaled, Y, test_size=0.2, random_state=42)

selector = SelectKBest(score_func=f_regression, k=6)
X_train_new = selector.fit_transform(X_train, y_train)
best_feature_indices = selector.get_support(indices=True) 
original_column_names = X.columns.to_list()  # Store the names
best_features = np.array(original_column_names)[best_feature_indices]


df_corr = pd.concat([X, Y], axis=1, ignore_index=False)
corr_matrix = df_corr.corr()

st.header("House Prices Predictor")
st.write("""
Data Source:  [Github](https://github.com/rasbt/machine-learning-book/blob/main/ch09/AmesHousing.txt)       

Author: Domenick Dobbs          
""")
st.divider()
st.image("Pictures/Ames_Downtown.png")
st.caption("Downtown Ames, Iowa in the summer.")
st.caption("Source: https://www.worldatlas.com/cities/ames-iowa.html")
st.divider()
st.write("""
## Data Background

The Ames Housing Dataset was compiled by Dean De Cock (Iowa State University) in 2011 for use in research and education.
The data captures information on residential home sales in Ames, Iowa between 2006 and 2010.
The Full dataset contains 2930 records and it is a commonly used dataset for Exploratory Data Analysis for Machine Learning Regression.     
    
---
         
## Goal 
         
The primary goal of this project is to build a predictive model that can reliably estimate the sale price of a house in Ames, Iowa. This model will leverage various housing attributes, like living area, number of bedrooms, and overall quality, to uncover patterns and make informed predictions.
         
---

""")

st.write("""
## Columns Used in Analysis

| Column Name | Data Type | Description |
|---|---|---|
| Lot Area | Continuous | Lot size (sq. ft.) |
| Overall Qual | Ordinal | Rates overall material and finish (1-10)| 
| Total Bsmt SF | Continuous | Total basement area (sq. ft.) |
| Gr Liv Area | Continuous | Above-grade living area (sq. ft.) |
| Full Bath | Discrete | Full bathrooms above grade |
| TotRms AbvGrd | Discrete | Total rooms above grade (excluding bathrooms) |
| Fireplaces | Discrete | Number of fireplaces |
| SalePrice | Continuous | Sale price ($) |

***
""")
xgb_regr_model = xgb.XGBRegressor()
xgb_regr_model.fit(X_train, y_train)


y_pred = xgb_regr_model.predict(X_test)
mse = metrics.mean_squared_error(y_test, y_pred)
mae = metrics.mean_absolute_error(y_test, y_pred)
r2 = metrics.r2_score(y_test, y_pred)


st.write(f"""
## Model Summary
- **Model Type**: I'm using an XGBoost Regressor model. This is a powerful type of gradient boosting algorithm that builds decision trees in an ensemble to make predictions. It's known for its accuracy and ability to handle a wide variety of data types.

- **Feature Scaling**: I've applied a MinMaxScaler to the data. This scaling technique helps ensure that all features in the dataset have a similar range (typically between 0 and 1), which can improve the performance of the model.
 
---
         
##  Model Performance Metrics
- Mean Squared Error: {mse:.2f}
- Mean Absolute Error : {mae:.2f}
- R-Squared: {r2:.2f}

 """)
st.divider()

with open('my_model.pkl', 'wb') as f:
    pickle.dump(xgb_regr_model, f) 

with open('scaler.pkl', 'wb') as f: 
    pickle.dump(Scalerminmax, f) 