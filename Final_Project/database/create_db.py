import pandas as pd
import sqlite3
import tensorflow as tf

# Load MNIST data
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Flatten images for easier handling
train_images = train_images.reshape((train_images.shape[0], -1))
test_images = test_images.reshape((test_images.shape[0], -1))

# Create DataFrames
train_df = pd.DataFrame(train_images, columns=[f"pixel_{i}" for i in range(784)])
train_df['label'] = train_labels

test_df = pd.DataFrame(test_images, columns=[f"pixel_{i}" for i in range(784)])
test_df['label'] = test_labels

mnist_df = pd.concat([train_df, test_df], ignore_index=True)

# 1. Load Data into DataFrame 
titanic_df = pd.read_csv('data_files/titanic_data.csv') 

columns = ['Overall Qual', 'Overall Cond', 'Gr Liv Area', 'Central Air', 'Total Bsmt SF', 'SalePrice', 'Lot Area', 'Full Bath', 'Half Bath', 'TotRms AbvGrd', 'Fireplaces', 'Wood Deck SF']
housing_df = pd.read_csv('data_files/AmesHousing.txt',
                 sep='\t',
                 usecols=columns)

movie_df = pd.read_csv("data_files/movie_data.csv")

# 2. Create or Connect to Database
conn = sqlite3.connect('RA_projects.db')  

# 3. Write DataFrame to Database 
titanic_df.to_sql('titanic', conn, if_exists='replace', index=False)
housing_df.to_sql('housing', conn, if_exists='replace', index=False)
movie_df.to_sql('movie', conn, if_exists='replace', index=False)
mnist_df.to_sql('mnist', conn, if_exists='replace', index=False)

# 4. Close Connection
conn.close()