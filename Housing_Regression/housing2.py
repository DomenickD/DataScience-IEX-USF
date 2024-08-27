import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, learning_curve, validation_curve
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from hyperopt import hp, tpe, fmin, Trials, STATUS_OK
from hyperopt.pyll.base import scope
import matplotlib.pyplot as plt

# Load Data
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
df = pd.read_csv("Housing_Regression\AmesHousing.txt", sep="\t", usecols=columns)
df["Central_Air_Binary"] = df["Central Air"].map({"N": 0, "Y": 1})

# Prepare Data
y = df["SalePrice"]
X = df.drop(["Central Air", "SalePrice"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define Pipelines
pipelines = {
    "xgb": Pipeline([("scaler", MinMaxScaler()), ("xgb", xgb.XGBRegressor())]),
    "rf": Pipeline([("scaler", MinMaxScaler()), ("rf", RandomForestRegressor())]),
}


# Define training function with curves
def train_and_evaluate_model(pipeline, X_train, y_train, X_test, y_test, model_name):
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    print(f"Results for {model_name} Model:")
    print(f"  MSE: {mean_squared_error(y_test, y_pred):.2f}")
    print(f"  MAE: {mean_absolute_error(y_test, y_pred):.2f}")
    print(f"  R2 Score: {r2_score(y_test, y_pred):.2f}")

    # Learning Curve
    train_sizes, train_scores, test_scores = learning_curve(
        pipeline,
        X_train,
        y_train,
        cv=5,
        scoring="r2",
        n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    plt.figure()
    plt.plot(train_sizes, train_scores_mean, "o-", color="r", label="Training score")
    plt.plot(
        train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
    )
    plt.title(f"Learning Curve for {model_name}")
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.legend(loc="best")
    plt.show()

    # Validation Curve (Example: n_estimators for RandomForest)
    param_range = (
        np.arange(10, 200, 10) if model_name == "rf" else np.logspace(-3, 0, 10)
    )
    param_name = "rf__n_estimators" if model_name == "rf" else "xgb__learning_rate"
    train_scores, test_scores = validation_curve(
        pipeline,
        X_train,
        y_train,
        param_name=param_name,
        param_range=param_range,
        cv=5,
        scoring="r2",
        n_jobs=-1,
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    plt.figure()
    plt.plot(param_range, train_scores_mean, "o-", color="r", label="Training score")
    plt.plot(
        param_range, test_scores_mean, "o-", color="g", label="Cross-validation score"
    )
    plt.title(f"Validation Curve for {model_name} ({param_name})")
    plt.xlabel(param_name)
    plt.ylabel("Score")
    plt.legend(loc="best")
    plt.show()


# Train and evaluate models
for model_key, pipeline in pipelines.items():
    train_and_evaluate_model(pipeline, X_train, y_train, X_test, y_test, model_key)
