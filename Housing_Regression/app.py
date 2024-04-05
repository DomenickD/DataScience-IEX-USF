#import modules
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler #During experimentation, this one helped alot
# from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

import seaborn as sns
from matplotlib import pyplot as plt

#build front end with streamlit
import streamlit as st # streamlit run Housing.py

from sklearn.feature_selection import SelectKBest, f_regression

from sklearn.model_selection import train_test_split as tts

from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import r2_score
from sklearn import metrics  # For evaluation metrics
import xgboost as xgb #pip install xgboost

st.header("House Prices Predictor")
st.write("""
Data Source:  [Github](https://github.com/rasbt/machine-learning-book/blob/main/ch09/AmesHousing.txt)       

Author: Domenick Dobbs 

***         
""")

# Given by the text
columns = ['Overall Qual', 'Overall Cond', 'Gr Liv Area',
           'Central Air', 'Total Bsmt SF', 'SalePrice', 'Lot Area', 
           'Full Bath', 'Half Bath', 'TotRms AbvGrd', 'Fireplaces', 'Wood Deck SF']

df = pd.read_csv('AmesHousing.txt',
                 sep='\t',
                 usecols=columns)


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


# Change Central Air to a binary variable
df["Central_Air_Binary"] = df['Central Air'].map({'N': 0, 'Y': 1})
df = df.drop("Central Air", axis = 1)

df = df.dropna(axis=0)
y = df.SalePrice
df = df.drop("SalePrice", axis = 1)
df.isnull().sum()

#import scalar function
MinMaxScaler = MinMaxScaler()
df_scaled = MinMaxScaler.fit_transform(df)

X_train, X_test, y_train, y_test = tts(df_scaled, y, test_size=0.2, random_state=42)

selector = SelectKBest(score_func=f_regression, k=6)
X_train_new = selector.fit_transform(X_train, y_train)
best_features = df.columns[selector.get_support()]

# data = pd.DataFrame({'Features': best_features, 
#                      'F-Regression Score': selector.scores_[:6]})


df_corr = pd.concat([df, y], axis=1, ignore_index=False)
corr_matrix = df_corr.corr()

# Given by the text
columns = ['Gr Liv Area', 'Total Bsmt SF', 'Full Bath', 'TotRms AbvGrd', 'Fireplaces', 'Lot Area', 'Overall Qual', 'SalePrice']


df = pd.read_csv('AmesHousing.txt',
                 sep='\t',
                 usecols=columns)

df = df.dropna(axis=0)
df.isnull().sum()

features = ['Gr Liv Area', 'Total Bsmt SF', 'Full Bath', 'TotRms AbvGrd', 'Fireplaces', 'Lot Area', 'Overall Qual']
X = df[features]
y = df['SalePrice']

X_train, X_test, y_train, y_test = tts(X, y, test_size=0.25, random_state=42)

#MAke a function for repeated metrics so they can all look uniform!!
st.write("""---""")
def evaluate_model(model, name):
    y_pred = model.predict(X_test)
    mse = metrics.mean_squared_error(y_test, y_pred)
    mae = metrics.mean_absolute_error(y_test, y_pred)
    r2 = metrics.r2_score(y_test, y_pred)

    # print(f"Model: {name}")
    # print(f"  MSE: {mse:.2f}")
    # print(f"  MAE: {mae:.2f}")
    # print(f"  R-squared: {r2:.2f}")
    # print("-----------------")

    st.write(f"Model: {name}")
    st.metric("MSE", f"{mse:.2f}")
    st.metric("MAE", f"{mae:.2f}")
    st.metric("R-squared", f"{r2:.2f}")
    st.text("-----------------")
  

# Define your models
lr = LinearRegression()
dt = DecisionTreeRegressor()
rf = RandomForestRegressor()
lasso = Lasso(alpha=0.1)
elasticnet = ElasticNet(alpha=0.01, l1_ratio=0.01)
xgb_regr_model = xgb.XGBRegressor()

#Train them allll
lr.fit(X_train, y_train)
dt.fit(X_train, y_train)
rf.fit(X_train, y_train)
lasso.fit(X_train, y_train)
elasticnet.fit(X_train, y_train)
xgb_regr_model.fit(X_train, y_train)

evaluate_model(lr, 'Linear Regression')
evaluate_model(dt, 'Decision Tree')
evaluate_model(rf, 'Random Forest')
evaluate_model(lasso, 'Lasso Regression')
evaluate_model(elasticnet, 'ElasticNet Regression')
evaluate_model(xgb_regr_model, 'XGB Regression')
