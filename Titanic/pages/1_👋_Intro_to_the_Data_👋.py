import streamlit as st

st.header("Data Introduction")

st.write("""
---       
### List of Column Names and what the values represent
         
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