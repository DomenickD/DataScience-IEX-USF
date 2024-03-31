import streamlit as st
import plotly.express as px #pip install plotly
import plotly.graph_objects as go

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd



titanic_data = pd.read_csv("titanic_data.csv")

st.header("Data Visualizations")

fig = px.histogram(titanic_data, x= "Survived")
# plt.title("Survival Distribution")
fig.update_layout(
    title="Survival Distribution",
    xaxis_range=[-0.5, 1.5],
    yaxis_range = [0, 1000],  # Adjust zoom level
    barmode='overlay',  # Overlay bars for better visibility with few categories
    bargap=0.05  # Add a small gap between bars
)
fig.update_traces(marker_color=['red', 'green'], selector=dict(type='histogram'))
st.plotly_chart(fig)

st.caption("""**This plot represents the distribution of survivors on board.**     
- Red represents those who did NOT survive. 
- Green represents those who survived.""")

# Survived by Pclass
fig = px.histogram(titanic_data, x="Survived", color="Pclass", barmode='group',
                   title="Survival Distribution by Passenger Class",
                   category_orders={"Pclass": [1, 2, 3]})  # Ensure class order
fig.update_layout(xaxis_range=[-0.5, 1.5], bargap=0.1) 
st.plotly_chart(fig)

# Survived by Sex
fig = go.Figure(data=[
    go.Bar(name='Female', x=['Survived', 'Not Survived'], y=titanic_data[titanic_data['Sex'] == 'female']['Survived'].value_counts()),
    go.Bar(name='Male', x=['Survived', 'Not Survived'], y=titanic_data[titanic_data['Sex'] == 'male']['Survived'].value_counts())
])
fig.update_layout(title="Survival Distribution by Sex", barmode='group', xaxis_range=[-0.5, 1.5], bargap=0.1)
st.plotly_chart(fig)

fig = px.histogram(titanic_data, x="Age")
fig.update_layout(title="Age Distribution") 
st.plotly_chart(fig) 



st.caption("This plot represents the distribution of ages onboard the Titanic upon it's demise.")


