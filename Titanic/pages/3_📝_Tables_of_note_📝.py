import streamlit as st
import pandas as pd

st.header("Data Tables YOU need to know")

titanic_data = pd.read_csv("titanic_data.csv")
titanic_data
