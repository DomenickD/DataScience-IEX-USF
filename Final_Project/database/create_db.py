"""Create the db file on start in case of file size issue when loading."""

import sqlite3
import pandas as pd

# 1. Load Data into DataFrame
# titanic_df = pd.read_csv('data_files/titanic_data.csv')
titanic_df = pd.read_csv(r"app/data_files/titanic_data.csv")


columns = [
    "Overall Qual",
    "Overall Cond",
    "Gr Liv Area",
    "Central Air",
    "Total Bsmt SF",
    "SalePrice",
    "Lot Area",
    "Full Bath",
    "Half Bath",
    "TotRms AbvGrd",
    "Fireplaces",
    "Wood Deck SF",
]
housing_df = pd.read_csv(
    r"C:/Users/Domenick Dobbs/Desktop/IEX/DataScience-IEX-USF/\
        Final_Project/database/data_files/AmesHousing.txt",
    sep="\t",
    usecols=columns,
)

movie_df = pd.read_csv(
    r"C:/Users/Domenick Dobbs/Desktop/IEX/DataScience-IEX-USF/\
        Final_Project/database/data_files/movie_data.csv"
)

# 2. Create or Connect to Database
conn = sqlite3.connect("RA_projects.db")

# 3. Write DataFrame to Database
titanic_df.to_sql("titanic", conn, if_exists="replace", index=False)
housing_df.to_sql("housing", conn, if_exists="replace", index=False)
movie_df.to_sql("movie", conn, if_exists="replace", index=False)
# mnist_df.to_sql('mnist', conn, if_exists='replace', index=False)

# 4. Close Connection
conn.close()
