# Import files
import csv
import sqlite3
import pandas as pd
import os


# Define the function (may add this to helper_functions.py for import later)
def csv_to_sqlite(csv_file, db_file, table_name):
    """Imports data from a CSV file into an SQLite database table."""
    if os.path.exists(db_file):
        print(f"Database {db_file} already exists. Skipping CSV to SQLite conversion.")
        return
    # Connect to the database
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # Open the CSV file
    with open(csv_file, "r") as file:
        reader = csv.reader(file)

        # Get the column headers from the first row
        headers = next(reader)

        # Create the SQL table (if it doesn't exist)
        create_table_sql = f"""CREATE TABLE IF NOT EXISTS {table_name} (
                                {', '.join([f'{header} TEXT' for header in headers])}
                              );"""
        cursor.execute(create_table_sql)

        # Insert data into the table
        insert_sql = (
            f"INSERT INTO {table_name} VALUES ({', '.join(['?' for _ in headers])})"
        )
        for row in reader:
            cursor.execute(insert_sql, row)

    # Commit the changes and close the connection
    conn.commit()
    conn.close()


####################
####################
####################
####################


def txt_to_sqlite(txt_file, db_file, table_name, delimiter=None):
    """Imports data from a text file into an SQLite database table.

    Args:
        txt_file: Path to the text file.
        db_file: Path to the SQLite database file.
        table_name: Name of the table to create/insert into.
        delimiter: Delimiter used in the text file (e.g., '\t' for tab-separated).
                   If None, assumes unstructured text.
    """
    if os.path.exists(db_file):
        print(f"Database {db_file} already exists. Skipping TXT to SQLite conversion.")
        return

    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # Drop the table if it already exists to avoid conflicts
    cursor.execute(f"DROP TABLE IF EXISTS {table_name}")

    if delimiter:  # Structured text (e.g., CSV-like)
        with open(txt_file, "r") as file:
            reader = csv.reader(file, delimiter=delimiter)
            headers = next(reader)
            create_table_sql = f"""
                CREATE TABLE {table_name} (
                    {', '.join([f'"{header}" TEXT' for header in headers])}
                );
            """
            cursor.execute(create_table_sql)

            insert_sql = (
                f"INSERT INTO {table_name} VALUES ({', '.join(['?' for _ in headers])})"
            )
            for row in reader:
                cursor.execute(insert_sql, row)

    else:  # Unstructured text
        create_table_sql = f"CREATE TABLE {table_name} (content TEXT);"
        cursor.execute(create_table_sql)

        with open(txt_file, "r") as file:
            for line in file:
                insert_sql = f"INSERT INTO {table_name} VALUES (?)"
                cursor.execute(insert_sql, (line.strip(),))

    conn.commit()
    conn.close()


####################
####################
####################
####################


# Define a function to execute a SELECT * statement and print the results in a tabular format
def select_all_from_table(db_file, table_name):
    """Executes a SELECT * statement on the specified table and prints the results in a tabular format."""
    conn = sqlite3.connect(db_file)
    query = f"SELECT * FROM {table_name} LIMIT 5"

    df = pd.read_sql_query(query, conn)

    print(f"\nData from table {table_name} in database {db_file}:\n")
    print(
        df.to_string(index=False)
    )  # Convert DataFrame to a string and print without the index

    conn.close()


####################
####################
####################
####################


# Define a function to load tab-separated data into a pandas DataFrame and print it
def load_tab_separated_to_dataframe(txt_file):
    """Loads tab-separated data from a text file into a pandas DataFrame and prints it."""
    df = pd.read_csv(txt_file, delimiter="\t")
    return df


# identify csv, db, and table
# Make these names your own!
csv_file_path = "CSV_Files/titanic_data.csv"
txt_file_path = "CSV_Files/AmesHousing.txt"

database_file_path_CSV = "DB_Files/titanic_data.db"
database_file_path_TXT = "DB_Files/AmesHousing.db"

table_name_CSV = "titanic_data"
table_name_TXT = "AmesHousing"

csv_to_sqlite(csv_file_path, database_file_path_CSV, table_name_CSV)
txt_to_sqlite(txt_file_path, database_file_path_TXT, table_name_TXT, delimiter="\t")

# Execute SELECT * statements
select_all_from_table(database_file_path_CSV, table_name_CSV)
select_all_from_table(database_file_path_TXT, table_name_TXT)

df_txt = load_tab_separated_to_dataframe(txt_file_path)
print(f"\nData from text file {txt_file_path}:\n")
print(df_txt.to_string(index=False))

conn = sqlite3.connect("DB_Files/titanic_data.db")
query = f"""
SELECT * 
FROM {table_name_CSV} 
WHERE Survived = 0 
LIMIT 5;
"""

df = pd.read_sql_query(query, conn)

print(df)

# Query to count the number of people who did not survive
query_count = f"SELECT COUNT(*) FROM {table_name_CSV} WHERE Survived = 0"

# Execute the query and fetch the result
count_result = pd.read_sql_query(query_count, conn)

# Display the count result
print(f"Number of people who did not survive: {count_result.iloc[0, 0]}")
