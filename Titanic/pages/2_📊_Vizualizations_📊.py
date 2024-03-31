import streamlit as st
import plotly.express as px #pip install plotly
import plotly.graph_objects as go

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np



titanic_data = pd.read_csv("titanic_data.csv")

st.header("Data Visualizations")


st.write("""---""")
st.subheader("Survival Distribution")
fig = px.histogram(titanic_data, x= "Survived")
fig.update_layout(
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

st.write("""---""")
st.subheader("Survival Distribution by Passenger Class")
fig = px.histogram(titanic_data, x="Survived", color="Pclass", barmode='group',
                   category_orders={"Pclass": [1, 2, 3]})
fig.update_layout(xaxis_range=[-0.5, 1.5], bargap=0.1) 
st.plotly_chart(fig)

st.caption("""something here""")

st.write("""---""")
st.subheader("Survival Distribution by Sex")
fig = go.Figure(data=[
    go.Bar(name='Female', x=['Survived', 'Not Survived'], y=titanic_data[titanic_data['Sex'] == 'female']['Survived'].value_counts()),
    go.Bar(name='Male', x=['Survived', 'Not Survived'], y=titanic_data[titanic_data['Sex'] == 'male']['Survived'].value_counts())
])
fig.update_layout(barmode='group', xaxis_range=[-0.5, 1.5], bargap=0.1)
st.plotly_chart(fig)

st.caption("""something here""")

st.write("""---""")
st.subheader("Age Distribution")
fig = px.histogram(titanic_data, x="Age")
st.plotly_chart(fig) 

st.caption("This plot represents the distribution of ages onboard the Titanic upon it's demise.")

st.write("""---""")
st.subheader("Titanic Heatmap")

titanic_numbers = titanic_data.select_dtypes(include=np.number)
titanic_numbers = titanic_numbers.drop("Unnamed: 0", axis=1)
mask = np.zeros_like(titanic_numbers.corr()) 
mask[np.triu_indices_from(mask)] = True

plt.figure(figsize=(8,6))
sns.heatmap(titanic_numbers.corr(), annot=True, cmap='coolwarm', mask=mask)
plt.xticks(rotation=45)
st.pyplot(plt.gcf()) 




