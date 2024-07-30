import streamlit as st
import requests
import pandas as pd

st.title('Database Query App')

#I am going to define functions for cleaner coding in my if statments for the submit button
def titanic_display():
    st.write("""
    # Titanic Classification Dataset""")
    st.divider()

    st.image("Pictures/Titanic.png")
    st.caption("Source: https://cdn.britannica.com/79/4679-050-BC127236/Titanic.jpg")
    st.divider()

    st.write("""
    ## Overview of this project from Kaggle
    The sinking of the Titanic is one of the most infamous shipwrecks in history.

    On April 15, 1912, during her maiden voyage, the widely considered “unsinkable” RMS Titanic sank after colliding with an iceberg. Unfortunately, there weren't enough lifeboats for everyone onboard, resulting in the death of 1502 out of 2224 passengers and crew.

    While there was some element of luck involved in surviving, it seems some groups of people were more likely to survive than others.

    In this challenge, we ask you to build a predictive model that answers the question: “what sorts of people were more likely to survive?” using passenger data (ie name, age, gender, socio-economic class, etc).
    """)

    st.write("""
    ---
    ## Disclaimer about the data
    This dataset does not include all of the PEOPLE from the actual Titanic. 
    There are 1309 rows of data for *passengers* in this Kaggle Dataset. There were 2240 total Passengers **and** Crew.
    As a result, the 931 crew members are not accounted for. 
    """)
    st.write("""
    ---
    ## Problem Statement
    The goal of this project is to develop a predictive model that accurately identifies factors influencing passenger survival rates during the tragic sinking of the RMS Titanic. 
            By analyzing historical passenger data, we seek to uncover patterns and relationships between individual characteristics 
            (such as age, gender, socio-economic class, cabin location, etc.) and their likelihood of survival.
            """)

    st.write("""
    ---       
    ## List of Column Names and what the values represent
            
    | Column Name    | Description                                                                     |
    |----------------|---------------------------------------------------------------------------------|
    | PassengerId    | A unique numerical identifier assigned to each passenger.                         |
    | Survived       | Survival status of the passenger (0 = No, 1 = Yes).                              |
    | Pclass         | The passenger's ticket class (1 = 1st Class, 2 = 2nd Class, 3 = 3rd Class).   |
    | Name           | The passenger's full name.                                                      |
    | Sex            | The passenger's gender (male, female).                                         |
    | Age            | The passenger's age in years. Fractional values may exist for younger children. |
    | SibSp          | The number of siblings or spouses traveling with the passenger.                   |
    | Parch          | The number of parents or children traveling with the passenger.                   |
    | Ticket         | The passenger's ticket number.                                                  |
    | Fare           | The price the passenger paid for their ticket.                                  |
    | Cabin          | The passenger's cabin number (if recorded).                                    |
    | Embarked       | The passenger's port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton). |
    ---
    """)

    st.subheader("Model Details")
    st.write("""
    This data was run against multiple models and multiple normalization methods. The highest ratings were from the logistic regression model with a standardized MinMaxScalar provided by Sci-kit learn.

    Model Accuracy before Hyperparameter Tuning:

    96.89%.

    Model Accuracy AFTER Hyperparameter Tuning:

    97.85%.

    """)

    st.divider()


    st.image("Pictures/Distro_by_sex.png")
    st.caption("This shows a side by side of the amount of males to females who survived the Titanic and did not survive.")

    st.divider()

    st.image("Pictures/Heatmap copy.png")
    st.caption("This Heatmap shows the correlation between features and 'Survived'.")

    st.divider()

    st.image("Pictures/Survival_Distro_by_Class.png")
    # st.caption("Survival status of the passenger (0 = No, 1 = Yes).")

    st.divider()

    st.image("Pictures/survival_dist.png")
    st.caption("""
This plot represents the distribution of survivors on board.

- Red represents those who did NOT survive.
               
- Green represents those who survived.""")

    st.divider()

    #learning curve
    st.image("Pictures/Learning_Curve copy.png")
    st.caption("This is the learning curve for the model.")

    st.divider()

    # DBSCAN
    st.image("Pictures/DBSCAN.png")
    st.caption("Here is the attempt at applying DBSCAN to the Titanic Dataset. No noticable trends.")

    st.divider()

def housing_display():
    st.header("The Ames Housing Data from Ames, Iowa")
    st.divider()
    st.image("Pictures/Ames_Downtown.png")
    st.divider()
    st.write("""
        ## Data Background

        The Ames Housing Dataset was compiled by Dean De Cock (Iowa State University) in 2011 for use in research and education.
        The data captures information on residential home sales in Ames, Iowa between 2006 and 2010.
        The Full dataset contains 2930 records and it is a commonly used dataset for Exploratory Data Analysis for Machine Learning Regression.     
            
        ---
                
        ## Goal 
                
        The primary goal of this project is to build a predictive model that can reliably estimate the sale price of a house in Ames, Iowa. This model will leverage various housing attributes, like living area, number of bedrooms, and overall quality, to uncover patterns and make informed predictions.
                
        ---

        """)
    st.subheader("Methodology")
    st.write("Approaching this dataset, we wanted to find out what type of Clusters were present so we could best see hwat features correlate with eachother.")
    st.write("To this end, we performed unsupervised learning as well as a correlation matrix as seen below.")
    st.image("Pictures/Elbow_Plot.png")
    st.caption("This shows the clusters. We call this an elbow plot because a well defined elbow is a roadmap to the optimal number of clusters vs. distortions.")
    st.divider()
    st.image("Pictures/Heatmap.png")
    st.caption("This is the heatmap. Our task is to predict Sale Price based on the data so we are only focused on what has a strong negative or positive correlation with the Sale Price. This is denoted by a number being closer to positive or negative 1.")
    st.divider()

    st.write("""
        ## Columns Used in Analysis

        | Column Name | Data Type | Description |
        |---|---|---|
        | Lot Area | Continuous | Lot size (sq. ft.) |
        | Overall Qual | Ordinal | Rates overall material and finish (1-10)| 
        | Total Bsmt SF | Continuous | Total basement area (sq. ft.) |
        | Gr Liv Area | Continuous | Above-grade living area (sq. ft.) |
        | Full Bath | Discrete | Full bathrooms above grade |
        | TotRms AbvGrd | Discrete | Total rooms above grade (excluding bathrooms) |
        | Fireplaces | Discrete | Number of fireplaces |
        | SalePrice | Continuous | Sale price ($) |

        ***
        """)
    st.write(f"""
        ## Model Summary
        - **Model Type**: I'm using an XGBoost Regressor model. This is a powerful type of gradient boosting algorithm that builds decision trees in an ensemble to make predictions. It's known for its accuracy and ability to handle a wide variety of data types.

        - **Feature Scaling**: I've applied a MinMaxScaler to the data. This scaling technique helps ensure that all features in the dataset have a similar range (typically between 0 and 1), which can improve the performance of the model.
        
        ---
                
        ##  Model Performance Metrics before Hyperparameter Tuning
        - Mean Squared Error: 914814901.49
        - Mean Absolute Error : 19989.89
        - R-Squared: 0.8772

        ##  Model Performance Metrics AFTER Hyperparameter Tuning
        - Mean Squared Error: 782524584.15
        - Mean Absolute Error : 18770.51
        - R-Squared: 0.8950

        """)
    st.divider()

    st.image("Pictures/relation_abvgrdliv_to_saleprice.png")
    st.caption("This helps to visualize if there's a positive correlation (and whether it's linear or not).")

    st.divider()
    st.image("Pictures/Learning_Curve.png")
    st.caption("The pattern shown on this raph shows a high variance.")
    

def movie_display():
    pass

# Radio buttons for single table selection
table_option = st.radio("Select dataset:", 
                       ("Titanic", "Housing", "Movie"))
                        # , "MNIST"
                        # ))

query = ""
if table_option == "Titanic":
    query = "SELECT * FROM titanic;"
elif table_option == "Housing":
    query = "SELECT * FROM housing;" 
elif table_option == "Movie":
    query = "SELECT * FROM movie LIMIT 50;"   
# elif table_option == "MNIST":
#     query = "SELECT * FROM mnist;"


query = st.text_area(label= 'Enter your SQL query here:', value = query)

# st.write(query)

if st.button('Submit'):
    response = requests.post('http://flask_route:5000/query', json={'query': query})

    if response.status_code == 200:
        try:
            result = response.json()
            data = result.get("data", [])
            columns = result.get("columns", [])
            # Convert the JSON response to a pandas DataFrame with column names
            df = pd.DataFrame(data, columns=columns)
            # st.dataframe(df)
            ### Think of a way to display results number that were retrived by the query

            ### Make Functions to display
            if table_option == "Titanic":
                df = df.drop(columns=['Unnamed: 0'])
                st.dataframe(df)
                counting = df["PassengerId"].count()
                st.write(f"{counting} results are displayed.")
                titanic_display()

            elif table_option == "Housing":
                st.dataframe(df)
                counting = df["SalePrice"].count()
                st.write(f"{counting} results are displayed.")
                housing_display()

            elif table_option == "Movie":
                st.dataframe(df)
                counting = df["review"].count()
                st.write(f"{counting} results are displayed.")
                movie_display()

        except requests.exceptions.JSONDecodeError:
            st.error("Error: The response is not in JSON format.")
            st.write("Response content:", response.text)

    else:
        st.error(f"Error: Received status code {response.status_code}")
        st.write("Response content:", response.text)
    












       
