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
df

st.write("""
## We will be using the following columns:
         
Overall Qual (Ordinal): Rates the overall material and finish of the house

    10	Very Excellent
    9	Excellent
    8	Very Good
    7	Good
    6	Above Average
    5	Average
    4	Below Average
    3	Fair
    2	Poor
    1	Very Poor
         
Overall Cond (Ordinal): Rates the overall condition of the house

    10	Very Excellent
    9	Excellent
    8	Very Good
    7	Good
    6	Above Average	
    5	Average
    4	Below Average	
    3	Fair
    2	Poor
    1	Very Poor
         
Gr Liv Area (Continuous): Above grade (ground) living area square feet
         
Central Air (Nominal): Central air conditioning

    N	No
    Y	Yes

Total Bsmt SF (Continuous): Total square feet of basement area

SalePrice (Continuous): Sale price $$
         
Lot Area (Continuous): Lot size in square feet
         
Full Bath (Discrete): Full bathrooms above grade
         
Half Bath (Discrete): Half baths above grade
         
TotRmsAbvGrd	(Discrete): Total rooms above grade (does not include bathrooms)

Fireplaces (Discrete): Number of fireplaces
         
Wood Deck SF (Continuous): Wood deck area in square feet
        
***
""")

st.write("""
# Data Preprocessing

The only data cleaning that we will do right now is convert the Central Air from Y and N to a binary.
        
        'N'= 0, 'Y'= 1 
         
***
""")

# Change Central Air to a binary variable
df["Central_Air_Binary"] = df['Central Air'].map({'N': 0, 'Y': 1})
df = df.drop("Central Air", axis = 1)
df
st.caption("""The last column is the new binary central air.""")
#Importing this from the book, just in case but I want to venture from the book this time (v3)
class LinearRegressionGD:
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.array([0.])
        self.losses_ = []

        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output)
            self.w_ += self.eta * 2.0 * X.T.dot(errors) / X.shape[0]
            self.b_ += self.eta * 2.0 * errors.mean()
            loss = (errors**2).mean()
            self.losses_.append(loss)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_) + self.b_

    def predict(self, X):
        return self.net_input(X)


code_LR = ("""
class LinearRegressionGD:
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.array([0.])
        self.losses_ = []

        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output)
            self.w_ += self.eta * 2.0 * X.T.dot(errors) / X.shape[0]
            self.b_ += self.eta * 2.0 * errors.mean()
            loss = (errors**2).mean()
            self.losses_.append(loss)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_) + self.b_

    def predict(self, X):
        return self.net_input(X)
""") 
st.code(code_LR, language='python')

st.caption("""This code block is an exert from the 'Machine Learning with Pytorch and Sci-Kit Learn Textbook'""")

df = df.dropna(axis=0)
y = df.SalePrice
df = df.drop("SalePrice", axis = 1)
df.isnull().sum()

st.write("""---""")

#import scalar function
MinMaxScaler = MinMaxScaler()
df_scaled = MinMaxScaler.fit_transform(df)

X_train, X_test, y_train, y_test = tts(df_scaled, y, test_size=0.2, random_state=42)

selector = SelectKBest(score_func=f_regression, k=6)
X_train_new = selector.fit_transform(X_train, y_train)
best_features = df.columns[selector.get_support()]

data = pd.DataFrame({'Features': best_features, 
                     'F-Regression Score': selector.scores_[:6]})
st.subheader("Feature Importance")
# Display the bar chart
st.bar_chart(data, x='Features', y='F-Regression Score', use_container_width=True) 
st.caption("""The F-test is a statistical test that helps assess whether a linear regression model, with multiple explanatory variables, provides a better fit to the data compared to a simpler model with no explanatory variables.""")


df_corr = pd.concat([df, y], axis=1, ignore_index=False)
corr_matrix = df_corr.corr()

st.write("""
---
### This Heatmap will display correlations between features.         
""")
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

# Create the interactive heatmap (with masking)
fig, ax = plt.subplots(figsize=(10, 8))  
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax, mask=mask) 
st.pyplot(fig)

st.caption("""We are only concerned with what correlates with the SalesPrice feature.""")

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

    print(f"Model: {name}")
    print(f"  MSE: {mse:.2f}")
    print(f"  MAE: {mae:.2f}")
    print(f"  R-squared: {r2:.2f}")
    print("-----------------")

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
