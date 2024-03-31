import streamlit as st
import pandas as pd
import os.path

st.set_page_config(
    page_title="Titanic Analysis",
    page_icon="👋")

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

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
gender_submission = pd.read_csv("gender_submission.csv")

train = pd.concat([train, pd.get_dummies(train["Pclass"], prefix='Pclass')], axis=1)
test = pd.concat([test, pd.get_dummies(test["Pclass"], prefix='Pclass')], axis=1)

train['Sex_binary'] = train.Sex.map({"male": 0, "female": 1}) 
test['Sex_binary'] = test.Sex.map({"male": 0, "female": 1})

train['Age'].fillna(value = round(train['Age'].mean()), inplace = True)
test['Age'].fillna(value = round(test['Age'].mean()), inplace = True) 

test['Fare'].fillna(value = round(test['Fare'].mean()), inplace = True) 


test_merged = pd.merge(test, gender_submission, how="inner")

titanic_data = pd.concat([train, test_merged], ignore_index=True)
# titanic_data = titanic_data.drop("Unnamed: 0", axis=1)

if not os.path.exists("titanic_data.csv"):
    titanic_data.to_csv("titanic_data.csv")

columns_to_drop = ["Pclass", "Name", "Sex",  "Ticket", "Cabin", "Embarked"]

train = train.drop(columns_to_drop, axis = 1)
test = test.drop(columns_to_drop, axis = 1)

