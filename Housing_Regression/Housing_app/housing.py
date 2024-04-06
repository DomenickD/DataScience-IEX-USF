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
MinMaxScaler = MinMaxScaler()
df_scaled = MinMaxScaler.fit_transform(X)

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

***         
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