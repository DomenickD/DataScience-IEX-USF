import streamlit as st
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder

st.header("Data Tables YOU need to know")

titanic_data = pd.read_csv("titanic_data.csv")
columns_to_drop = ["Unnamed: 0"] #, "Name", "Ticket", "Cabin", "Sex", "Embarked" 
titanic_data = titanic_data.drop(columns_to_drop, axis=1)
titanic_data

st.caption("This is the combined dataset for the Titanic Data. It included the Train and Test csv files from Kaggle.")




