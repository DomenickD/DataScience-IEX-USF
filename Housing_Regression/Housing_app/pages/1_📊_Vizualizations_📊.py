import streamlit as st

import matplotlib.pyplot as plt
import seaborn as sns
import plotly
import plotly.express as px
import numpy as np

import pickle
import Housing_Regression.Housing_app.housing as housing

st.header("Visualizations")

st.write("""---""")
st.subheader("Feature Importance")
fig = px.bar(x=housing.best_features, y=housing.selector.scores_[:6], 
             labels={'x': 'Features',
                      'y': 'F-Regression Score'},)
fig.update_layout(xaxis={'tickangle': 30}, 
                  xaxis_range=[-0.5, 5.5],
                  yaxis_range=[0, 4500], 
                  bargap=0.2)  # Adjust tick label rotation
st.plotly_chart(fig)
st.caption("""Something Here""")
st.write("""---""")

# fig, ax = plt.subplots(figsize=(11, 8))  
# sns.heatmap(housing.corr_matrix, mask=housing.mask, cmap='coolwarm', vmin=-1, vmax=1, annot=True, ax=ax)
# plt.xticks(rotation=45)

# ax.set_title('Correlation Heatmap')
# plt.show()

st.write("""
### This Heatmap will display correlations between features.         
""")
mask = np.triu(np.ones_like(housing.corr_matrix, dtype=bool))

# Create the interactive heatmap (with masking)
fig, ax = plt.subplots(figsize=(10, 8))  
sns.heatmap(housing.corr_matrix, annot=True, cmap='coolwarm', ax=ax, mask=mask) 
st.pyplot(fig)

st.caption("""We are only concerned with what correlates with the SalesPrice feature.""")