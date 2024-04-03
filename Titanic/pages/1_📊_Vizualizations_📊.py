import streamlit as st
import plotly.express as px #pip install plotly
import plotly.graph_objects as go

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np



titanic_data = pd.read_csv("titanic_data.csv")

st.header("The Data")
st.divider()
st.subheader("The Combined Titanic Dataframe")
titanic_data = pd.read_csv("titanic_data.csv")
columns_to_drop = ["Unnamed: 0"] #, "Name", "Ticket", "Cabin", "Sex", "Embarked" 
titanic_data = titanic_data.drop(columns_to_drop, axis=1)
titanic_data
st.caption("This is the combined dataset for the Titanic Data. It included the Train and Test csv files from Kaggle.")


st.write("""---""")
st.subheader("Survival Distribution")
# fig = px.histogram(titanic_data, x= "Survived")
# fig.update_layout(
#     xaxis_range=[-0.5, 1.5],
#     yaxis_range = [0, 1000],  # Adjust zoom level
#     barmode='overlay',  # Overlay bars for better visibility with few categories
#     bargap=0.05  # Add a small gap between bars
# )
# fig.update_traces(marker_color=['red', 'green'], selector=dict(type='histogram'))
# st.plotly_chart(fig)

survived_counts = titanic_data['Survived'].value_counts().sort_index()
fig = px.pie(survived_counts, 
            values=survived_counts.values, 
            names=['Did Not Survive', 'Survived'],
            color_discrete_sequence=['green', 'red'] # Optional: Adjust colors
            )

st.plotly_chart(fig) 

st.caption("""**This plot represents the distribution of survivors on board.**     
- Red represents those who did NOT survive. 
- Green represents those who survived.""")

st.write("""---""")
st.subheader("Survival Distribution by Passenger Class")

fig = px.violin(titanic_data, 
                y="Survived",  # Variable of interest
                x="Pclass",   # Categorical variable for splitting the violins
                color="Pclass",  # Color violins by category
                box=True,        # Show box plot elements within the violin
                points="all"    # Show all individual data points
               )

fig.update_layout(xaxis_title="Passenger Class") 
st.plotly_chart(fig) 
st.caption("""Survival status of the passenger (0 = No, 1 = Yes).""")

# fig = px.histogram(titanic_data, x="Survived", color="Pclass", barmode='group',
#                    category_orders={"Pclass": [1, 2, 3]})
# fig.update_layout(xaxis_range=[-0.5, 1.5], bargap=0.1) 
# st.plotly_chart(fig)
# st.caption("""something here""")

st.write("""---""")
st.subheader("Survival Distribution by Sex")

fig = go.Figure(data=[
    go.Bar(name='Female', x=['Survived', 'Not Survived'], 
           y=titanic_data[titanic_data['Sex'] == 'female']['Survived'].value_counts(),
           marker_color='pink'),
    go.Bar(name='Male', x=['Survived', 'Not Survived'], 
           y=titanic_data[titanic_data['Sex'] == 'male']['Survived'].value_counts(),
           marker_color='teal')
])
fig.update_layout(barmode='group', xaxis_range=[-0.5, 1.5], bargap=0.1)
st.plotly_chart(fig)

st.caption("""This shows a side by side of the amount of males to females who survived the Titanic and did not survive.""")

st.write("""---""")

st.subheader("Age Distribution (Grouped)")
fig = px.histogram(titanic_data, x="Age", nbins=9, range_x=[0, 80])  # Adjust range as needed
fig.update_layout(
    xaxis_title="Age (Groups of 10)",
    bargap=0.05,
    xaxis_range=[-5, 85],
    yaxis_range = [0, 600]
)
fig.update_traces(marker_color='lightgreen')
st.plotly_chart(fig) 


st.caption("This plot represents the distribution of ages onboard the Titanic.")

st.write("""---""")
st.subheader("Titanic Heatmap")

titanic_numbers = titanic_data.select_dtypes(include=np.number)
mask = np.zeros_like(titanic_numbers.corr()) 
mask[np.triu_indices_from(mask)] = True

plt.figure(figsize=(8,6))
sns.heatmap(titanic_numbers.corr(), annot=True, cmap='coolwarm', mask=mask)
plt.xticks(rotation=45)
st.pyplot(plt.gcf()) 






