import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import Housing_App as housing

st.header("Visualizations")

st.write("""---""")

column_to_filter_by = st.selectbox("Choose a column to filter by", housing.df.columns)
filter_options = st.multiselect("Filter by", options=housing.df[column_to_filter_by].unique())

# Filtering data based on selection
if filter_options:
    filtered_data = housing.df[housing.df[column_to_filter_by].isin(filter_options)]
else:
    filtered_data = housing.df

st.dataframe(filtered_data)
st.write(f"{filtered_data["Lot Area"].count()} results are displayed.")

st.divider()


st.subheader("Feature Importance")
fig = px.bar(x=housing.best_features, y=housing.selector.scores_[:6], 
             labels={'x': 'Features',
                      'y': 'F-Regression Score'})
fig.update_layout(xaxis={'tickangle': 30}, 
                  xaxis_range=[-0.5, 5.5],# Adjust tick label rotation
                  yaxis_range=[0, 4500], # Adjust tick label rotation
                  bargap=0.2)  
st.plotly_chart(fig)
st.caption("""Using an algorthim, we were able to determine the top 6 features.""")
st.divider()

###
st.subheader("Distribution of Sale Prices")
fig = px.histogram(housing.df, x="SalePrice", color_discrete_sequence=px.colors.qualitative.Light24_r)
fig.update_layout(xaxis_title="Sale Price ($)", yaxis_title="Count")
st.plotly_chart(fig)
st.caption("This helps us visually understand the overall shape of sale price distribution.")
st.divider()
###

###
st.subheader("Relationship between Above Grade Living Area and Sale Price")
fig = px.scatter(housing.df, x="Gr Liv Area", y="SalePrice", trendline="ols",
                 color="Overall Qual", color_discrete_map={"1": "blue", "5": "orange", "10": "green"})
fig.update_layout(xaxis_title="Above Grade Living Area (sq.ft.)", yaxis_title="Sale Price ($)")
st.plotly_chart(fig)
st.caption("This helps to visualize if there's a positive correlation (and whether it's linear or not).")
st.divider()
###
###
st.subheader("Sale Price Box Plots by Overall Quality")
fig = px.box(housing.df, x="Overall Qual", y="SalePrice", 
             color_discrete_sequence=px.colors.qualitative.Light24)
fig.update_layout(xaxis_title="Overall Quality", yaxis_title="Sale Price ($)")
st.plotly_chart(fig)
st.caption("See how the distribution and median sale prices differ across quality ratings. This shows that some the highest quality homes can be less than $200k.")
st.divider()
###

st.subheader("Correlation Heatmap")
# mask = np.triu(np.ones_like(housing.corr_matrix, dtype=bool))
# Create the interactive heatmap (with masking)
fig, ax = plt.subplots(figsize=(10, 8))  
# sns.heatmap(housing.corr_matrix, annot=True, cmap='coolwarm', ax=ax, mask=mask) 
sns.heatmap(housing.corr_matrix, annot=True, cmap='coolwarm', ax=ax) 
plt.xticks(rotation=45)
plt.yticks(rotation=45)
st.pyplot(fig)
st.caption("""This Heatmap will display correlations between features. We are only concerned with what correlates with the SalesPrice feature.""")