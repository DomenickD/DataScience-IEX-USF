"""
To use this file and it's functions, use the following code:
```
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
```
from helper_functions import [whatever function you want to use today!]

"""

from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import StandardScaler
import pprint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier


def train_and_evaluate_classification_model(
    X_train, y_train, X_test, y_test, models, scaler=StandardScaler()
):
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

        dict_key = f"{model_name} -- {scaler_name}"

        results[dict_key] = accuracy

    pprint.pprint(results)  # Pretty print the results dictionary - inspired by Brett
    return results


def train_and_evaluate_regression_model(
    X_train, y_train, X_test, y_test, models, scaler=StandardScaler()
):
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

        dict_key = f"{model_name} -- {scaler_name}"

        results[dict_key] = r2

    pprint.pprint(results)  # Pretty print the results dictionary - inspired by Brett
    return results


# Function for Logistic Regression hyperparameter tuning
def tune_logistic_regression(X_train, y_train):
    param_grid = {
        "C": [0.001, 0.01, 0.1, 1, 10, 100],  # Inverse of regularization strength
        "penalty": ["l1", "l2"],  # Regularization type (Lasso or Ridge)
        "solver": ["liblinear", "saga"],  # Solver algorithm
    }
    grid_search = RandomizedSearchCV(
        LogisticRegression(), param_grid, cv=5, scoring="accuracy"
    )
    grid_search.fit(X_train, y_train)
    return grid_search.best_params_


# Function for Gaussian Naive Bayes hyperparameter tuning
def tune_gaussian_nb(X_train, y_train):
    param_grid = {
        "var_smoothing": [
            1e-9,
            1e-8,
            1e-7,
            1e-6,
            1e-5,
        ]  # Smoothing parameter for variance
    }
    grid_search = RandomizedSearchCV(GaussianNB(), param_grid, cv=5, scoring="accuracy")
    grid_search.fit(X_train, y_train)
    return grid_search.best_params_


# Function for LinearSVC hyperparameter tuning
def tune_linear_svc(X_train, y_train):
    param_grid = {
        "C": [0.001, 0.01, 0.1, 1, 10, 100],  # Inverse of regularization strength
        "loss": ["hinge", "squared_hinge"],  # Loss function
        "dual": [
            True,
            False,
        ],  # Whether to solve the dual or primal optimization problem
    }
    grid_search = RandomizedSearchCV(
        LinearSVC(random_state=42), param_grid, cv=5, scoring="accuracy"
    )
    grid_search.fit(X_train, y_train)
    return grid_search.best_params_


# Function for Random Forest Classifier hyperparameter tuning
def tune_random_forest(X_train, y_train):
    param_grid = {
        "n_estimators": [100, 200, 300],  # Number of trees in the forest
        "max_depth": [None, 5, 10, 20],  # Maximum depth of the tree
        "min_samples_split": [
            2,
            5,
            10,
        ],  # Minimum number of samples required to split an internal node
        "min_samples_leaf": [
            1,
            2,
            4,
        ],  # Minimum number of samples required to be at a leaf node
    }
    grid_search = RandomizedSearchCV(
        RandomForestClassifier(random_state=42), param_grid, cv=5, scoring="accuracy"
    )
    grid_search.fit(X_train, y_train)
    return grid_search.best_params_
