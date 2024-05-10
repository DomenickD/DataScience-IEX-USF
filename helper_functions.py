"""
To use this file and it's functions, use the following code.

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
        models: A list of sklearn models to evaluate. Defaults to None (StandardScaler will be used).
        scaler: An sklearn scaler object to use. Defaults to StandardScaler.

    Returns:
        dict: A dictionary containing the accuracy scores for each model.
    """

    results = {}  # Dictionary to store results

    for model in models:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        if isinstance(model, StandardScaler):
            model_name = "StandardScaler"
        else:
            model_name = type(model).__name__

        results[model_name] = accuracy

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
        models: A list of sklearn models to evaluate. Defaults to None (StandardScaler will be used).
        scaler: An sklearn scaler object to use. Defaults to StandardScaler.

    Returns:
        dict: A dictionary containing the r2 scores for each model.
    """

    results = {}  # Dictionary to store results

    for model in models:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)

        if isinstance(model, StandardScaler):
            model_name = "StandardScaler"
        else:
            model_name = type(model).__name__

        results[model_name] = r2

    pprint.pprint(results)  # Pretty print the results dictionary - inspired by Brett
    return results 