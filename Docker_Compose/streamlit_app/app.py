import streamlit as st
import requests
import pandas as pd

st.title("Database Query App")

# Checkbox for selecting the table name
table_selected = st.checkbox("Use Titanic dataset", value=False)

query = ""
if table_selected:
    query = "SELECT * FROM titanic_data;"


query = st.text_area(label="Enter your SQL query here:", value=query)

# st.write(query)

if st.button("Submit"):
    response = requests.post("http://flask-container:5000/query", json={"query": query})

    if response.status_code == 200:
        try:
            result = response.json()
            data = result.get("data", [])
            columns = result.get("columns", [])
            # Convert the JSON response to a pandas DataFrame with column names
            df = pd.DataFrame(data, columns=columns)
            st.dataframe(df)
        except requests.exceptions.JSONDecodeError:
            st.error("Error: The response is not in JSON format.")
            st.write("Response content:", response.text)
    else:
        st.error(f"Error: Received status code {response.status_code}")
        st.write("Response content:", response.text)
