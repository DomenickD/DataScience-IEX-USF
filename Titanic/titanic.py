#import data minuplations modules
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline

#import viuslaization models
import matplotlib.pyplot as plt
import seaborn as sns

#import normalization modules
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

#import machine learning models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.svm import SVC

#import NN models - Pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

#import NN models - Tensorflow
import tensorflow as tf
# from tensorflow.keras import layers, models

#import accuracy_score function from sklearn.metrics to score models better
from sklearn.metrics import accuracy_score

import streamlit as st

st.write("""
# Titanic Classification Dataset
**Data Source:** [Kaggle](https://www.https://www.kaggle.com/c/titanic/data)
         
**Author:** Domenick Dobbs
***
## Overview of this project from Kaggle
The sinking of the Titanic is one of the most infamous shipwrecks in history.

On April 15, 1912, during her maiden voyage, the widely considered “unsinkable” RMS Titanic sank after colliding with an iceberg. Unfortunately, there weren't enough lifeboats for everyone onboard, resulting in the death of 1502 out of 2224 passengers and crew.

While there was some element of luck involved in surviving, it seems some groups of people were more likely to survive than others.

In this challenge, we ask you to build a predictive model that answers the question: “what sorts of people were more likely to survive?” using passenger data (ie name, age, gender, socio-economic class, etc).
""")
#load the data into dataframes
#data retrieved from kaggle competitions
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
gender_submission = pd.read_csv("gender_submission.csv")

st.write("""
***
These are the initial Datasets before any cleaning
""")

st.header("Train Dataset")
train

st.header("Test Dataset")
test

st.header("\"Gender_Submission\" Dataset")
gender_submission

st.write("""
***
""")

st.header("Data Cleaning")

st.write("""
Summary:
- Create Dummy Variables to represent Pclass in test and training dataset as binary
- Convert Male and femail to binary values (0=Male, 1=Female)
- Drop columns that are not needed such as "Pclass", "Name", "Sex",  "Ticket", "Cabin", "Embarked"
- Fill in missing values in the "Age" column for both test and train dataframes by replacing missing values with the mean of ages
- The test Dataset has some missing values in the "Fare" Column. Replace with mean for the column     
         
""")
#Dummies variables for PClass to prevent bias toward one number being weighed more than another
train = pd.concat([train, pd.get_dummies(train["Pclass"], prefix='Pclass')], axis=1)
test = pd.concat([test, pd.get_dummies(test["Pclass"], prefix='Pclass')], axis=1)

# Let's make the sex cloumn into a binary column
train['Sex_binary'] = train.Sex.map({"male": 0, "female": 1}) 
test['Sex_binary'] = test.Sex.map({"male": 0, "female": 1})

# columns_to_drop = ["Pclass", "Name", "Sex", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"]
columns_to_drop = ["Pclass", "Name", "Sex",  "Ticket", "Cabin", "Embarked"]

train = train.drop(columns_to_drop, axis = 1)
test = test.drop(columns_to_drop, axis = 1)

#Let's fill in the ages with the mean of all ages.
train['Age'].fillna(value = round(train['Age'].mean()), inplace = True) #look up .fillna function
test['Age'].fillna(value = round(test['Age'].mean()), inplace = True) 
train["Age"].count() #now we have every row accounted for. 

# test["Fare"].dropna(axis=0, how='any', inplace=True)
test['Fare'].fillna(value = round(test['Fare'].mean()), inplace = True) 

st.header("Train Dataset")
train

st.header("Test Dataset")
test

st.header("\"Gender_Submission\" Dataset")
gender_submission

st.write("""
***
""")

st.header("Data Organization (Replacing train_test_split)")

#I want to focus on training a model on Age, Sex_binary, FirstClass, SecondClass, ThirdClass, "SibSp", "Parch", "Fare"
#The goal is to predict whether or not the user survived based on this. 
train_features = train[["Age", "Sex_binary", "Pclass_1","Pclass_2", "Pclass_3", "Fare"]]
train_labels = train["Survived"]
test_features = test[["Age", "Sex_binary", "Pclass_1", "Pclass_2", "Pclass_3", "Fare"]]
test_labels = gender_submission["Survived"]

st.write("""
Summary:
- We are assigning the columns "Age", "Sex_binary", "Pclass_1","Pclass_2", "Pclass_3", "Fare" from the train dataframe to "train_features" 
- We are assigning the columns "Age", "Sex_binary", "Pclass_1","Pclass_2", "Pclass_3", "Fare" from the test dataframe to "test_features" 
- We are taking the "Survived" Column from the train dataframe and assigning it to the "train_labels" variable.    
- We are taking the "Survived" Column from the gender_submission dataframe and assigning it to the "test_labels" variable.    

By doing this, we are effectively taking 3 csv files and performing our own train_test_split on them so we can train our models and then test our models based on the partitioned dataframes.          
""")
#initialize an accuracy test key = model , value = accuracy score
model_accuracy_titanic_compare = {}

def train_and_evaluate_model(model, scaler, train_features, train_labels, test_features, test_labels):
    train_features_scaled = scaler.fit_transform(train_features)
    test_features_scaled = scaler.transform(test_features)
    
    model.fit(train_features_scaled, train_labels)
    y_predict = model.predict(test_features_scaled)
    
    accuracy = accuracy_score(test_labels, y_predict)
    
    model_key = f"{model.__class__.__name__} - {scaler.__class__.__name__}"
    model_accuracy_titanic_compare[model_key] = accuracy
    
    return accuracy

st.write("""
***
""")

st.header("Preparing for model training/fitting")

st.write("""
Summary:
- We will use a StandardScaler from sklearn.preprocessing to scale our features. This ensures that all of our features are centered.
- We will also use MinMaxScalar and RobustScalar. We are going to see if any scaling module gives us a better performance than others.
         
After that we will initialize the following models:
- LogisticRegression
- DecisionTreeClassifier
- RandomForestClassifier
- GaussianNB
- KNeighborsClassifier with 3 neighbors
- SVC - A support vector machine classifer

""")

# Initialize Scalar Models
scaler = StandardScaler()
min_max_scaler = MinMaxScaler()
robust_scaler = RobustScaler()

#initialze Logistic regression Models models
model = LogisticRegression()
model2 = LogisticRegression()
model3 = LogisticRegression()

# Initialize DecisionTreeClassifier Models
tree_model1 = DecisionTreeClassifier(criterion='entropy', max_depth=3)
tree_model2 = DecisionTreeClassifier(criterion='entropy', max_depth=3)
tree_model3 = DecisionTreeClassifier(criterion='entropy', max_depth=3)

# Initilize the RandomForest Classifiers 
RFC_model1 = RandomForestClassifier(n_estimators=80, criterion='gini', max_depth=4)
RFC_model2 = RandomForestClassifier(n_estimators=80, criterion='gini', max_depth=4)
RFC_model3 = RandomForestClassifier(n_estimators=80, criterion='gini', max_depth=4)

#initialize naive bayes
nb_model1 = GaussianNB()
nb_model2 = GaussianNB()
nb_model3 = GaussianNB()

# Initialize KNN
knn_model1 = KNeighborsClassifier(n_neighbors=3) 
knn_model2 = KNeighborsClassifier(n_neighbors=3) 
knn_model3 = KNeighborsClassifier(n_neighbors=3) 

# Initilize Support Vector Machines
svm_svc_model1 = svm.SVC(kernel='poly', C=1.5) 
svm_svc_model2 = svm.SVC(kernel='poly', C=1.5) 
svm_svc_model3 = svm.SVC(kernel='poly', C=1.5) 

# Train and evaluate models
LR_model_acc_score_Scalar = train_and_evaluate_model(model, scaler, train_features, train_labels, test_features, test_labels)
LR_model_acc_score_MinMax = train_and_evaluate_model(model2, min_max_scaler, train_features, train_labels, test_features, test_labels)
LR_model_acc_score_Robust = train_and_evaluate_model(model3, robust_scaler, train_features, train_labels, test_features, test_labels)

DecisionTree_acc_score_Scalar = train_and_evaluate_model(tree_model1, scaler, train_features, train_labels, test_features, test_labels)
DecisionTree_acc_score_MinMax = train_and_evaluate_model(tree_model2, min_max_scaler, train_features, train_labels, test_features, test_labels)
DecisionTree_acc_score_Robust = train_and_evaluate_model(tree_model3, robust_scaler, train_features, train_labels, test_features, test_labels)

RandomForest_acc_score_Scalar = train_and_evaluate_model(RFC_model1, scaler, train_features, train_labels, test_features, test_labels)
RandomForest_acc_score_MinMax = train_and_evaluate_model(RFC_model2, min_max_scaler, train_features, train_labels, test_features, test_labels)
RandomForest_acc_score_Robust = train_and_evaluate_model(RFC_model3, robust_scaler, train_features, train_labels, test_features, test_labels)

NB_acc_score_Scalar = train_and_evaluate_model(nb_model1, scaler, train_features, train_labels, test_features, test_labels)
NB_acc_score_MinMax = train_and_evaluate_model(nb_model2, min_max_scaler, train_features, train_labels, test_features, test_labels)
NB_acc_score_Robust = train_and_evaluate_model(nb_model3, robust_scaler, train_features, train_labels, test_features, test_labels)

KNN_acc_score_Scalar = train_and_evaluate_model(knn_model1, scaler, train_features, train_labels, test_features, test_labels)
KNN_acc_score_MinMax = train_and_evaluate_model(knn_model2, min_max_scaler, train_features, train_labels, test_features, test_labels)
KNN_acc_score_Robust = train_and_evaluate_model(knn_model3, robust_scaler, train_features, train_labels, test_features, test_labels)

SVC_acc_score_Scalar = train_and_evaluate_model(svm_svc_model1, scaler, train_features, train_labels, test_features, test_labels)
SVC_acc_score_MinMax = train_and_evaluate_model(svm_svc_model2, min_max_scaler, train_features, train_labels, test_features, test_labels)
SVC_acc_score_Robust = train_and_evaluate_model(svm_svc_model3, robust_scaler, train_features, train_labels, test_features, test_labels)

st.write("""
***
""")

st.header("Train and score models")

# print it all
print(model_accuracy_titanic_compare)
model_accuracy_titanic_compare
max_key_value_pair = max(model_accuracy_titanic_compare.items(), key=lambda x: x[1])
print(max_key_value_pair)
st.write("""
## The highest accuracy model is:
""")
max_key_value_pair

#I remember when I did this 2 years ago, we used Jack and Rose then ourselves to make predictions on the model and we mad ethem in a np.array
Jack = np.array([20.0, 0.0, 0.0, 0.0, 1.0, 8.0500])
Rose = np.array([17.0, 1.0, 1.0, 0.0, 0.0, 71.2833])
Dom = np.array([29.0,  0.0, 0.0, 1.0, 0.0, 30.0708])

passenger_predict = np.array([Jack, Rose, Dom])

passenger_predict = scaler.transform(passenger_predict)

st.write("""
***
# Predict
""")

# "Age", "Sex_binary", "Pclass_1","Pclass_2", "Pclass_3", "Fare"
st.write("""
Let's Input some people like some new data points to predict if they will survive the Titanic disaster.
- Jack is a 20 year old Male, third class passenger who paid $8.05 for his ticket.
- Rose is a 17 year old Female, first class passenger who paid $71.28 for her ticket.
- Dom is a 29 year old Male in second class passenger who paid $30.07 for his ticket.
***
""")


#prediction time! My favorite part
# Make survival predictions!
pass_predict = model.predict(passenger_predict) #This will print a 1 or 0 for surivied or did not survive 
print(pass_predict)
pass_predict #This will print a 1 or 0 for surivied or did not survive 
predict_proba = model.predict_proba(passenger_predict) #this will give us how likely for each option
print(predict_proba)
predict_proba #this will give us how likely for each option

st.write("""

- The first row is Jack. Jack had an 88.5% of NOT surviving based on the data. 
- The second row is Rose. Rose had a 95% chance of surviving. 
- The last row is Dom. Dom would of had a 76% chance of NOT surviving.
***
""")