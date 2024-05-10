"""
To use this file and it's functions, use the following code:

import requests
from pathlib import Path
# download helper functions from the github repo
if Path("helper_functions.py").is_file():
  print("Helper function exists, skipping download")
else:
  print("Download helper_functions.py")
  request = requests.get("https://raw.githubusercontent.com/DomenickD/DataScience-IEX-USF/main/helper_functions.py")
  with open("helper_functions.py", "wb") as f:
    f.write(request.content)

from helper_functions import [whatever function you want to use today!]

"""
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import StandardScaler
import pprint

def train_and_evaluate_classification_model(X_train, y_train, X_test, y_test, models, scaler=StandardScaler()):

    """
    Evaluates the performance of multiple machine learning models.

    Args:
        X_train: Training data features.
        y_train: Training data labels.
        X_test: Testing data features.
        y_test: Testing data labels.
        models: A list of sklearn models to evaluate. 
        scaler: An sklearn scaler object to use. Defaults to StandardScaler.

    Returns:
        dict: A dictionary containing the accuracy scores for each model.
    """

    results = {}  # Dictionary to store results

    for model in models:
        scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)

        scaler_name = type(scaler).__name__
        
        model_name = type(model).__name__

        dict_key = (f"{model_name} -- {scaler_name}")

        results[dict_key] = accuracy

    pprint.pprint(results)  # Pretty print the results dictionary - inspired by Brett
    return results 

def train_and_evaluate_regression_model(X_train, y_train, X_test, y_test, models, scaler=StandardScaler()):

    """
    Evaluates the performance of multiple machine learning models.

    Args:
        X_train: Training data features.
        y_train: Training data labels.
        X_test: Testing data features.
        y_test: Testing data labels.
        models: A list of sklearn models to evaluate. 
        scaler: An sklearn scaler object to use. Defaults to StandardScaler.

    Returns:
        dict: A dictionary containing the r2 scores for each model.
    """

    results = {}  # Dictionary to store results

    for model in models:
        scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        r2 = r2_score(y_test, y_pred)

        scaler_name = type(scaler).__name__
        
        model_name = type(model).__name__

        dict_key = (f"{model_name} -- {scaler_name}")

        results[dict_key] = r2

    pprint.pprint(results)  # Pretty print the results dictionary - inspired by Brett
    return results 