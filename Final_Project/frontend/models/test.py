import pickle

def load_and_print_model_details(pickle_file_path):
    """Loads a pickle file and prints details about the contained model."""

    with open(pickle_file_path, "rb") as f:
        model = pickle.load(f)

    print("\nModel Details:")
    print("-------------")
    print(f"Type: {type(model)}")  
    print(f"Model Class: {model.__class__.__name__}") 

    # If scikit-learn model:
    if hasattr(model, "get_params"):
        print(f"Parameters:\n{model.get_params()}")

    # If other model types, adapt based on their specific attributes/methods
    
    # Print more details specific to your model type (e.g., accuracy, features) if needed
    # ...

# Example usage
pickle_file_path = "xgb_pipeline_minmaxscaler.pkl"  # Replace with your file path
load_and_print_model_details(pickle_file_path)

"""Model Details:
-------------
Type: <class 'sklearn.linear_model._logistic.LogisticRegression'>
Model Class: LogisticRegression
Parameters:
{'C': 1.0, 'class_weight': None, 'dual': False, 'fit_intercept': True, 'intercept_scaling': 1, 'l1_ratio': None, 'max_iter': 100, 'multi_class': 'auto', 'n_jobs': None, 'penalty': 'l2', 'random_state': None, 'solver': 'lbfgs', 'tol': 0.0001, 'verbose': 0, 'warm_start': False}
PS C:\Users\Domenick Dobbs\Desktop\IEX\DataScience-IEX-USF\Final_Project\frontend\pickle> python test.py

Model Details:
-------------
Type: <class 'sklearn.pipeline.Pipeline'>
Model Class: Pipeline
Parameters:
{'memory': None, 'steps': [('scaler', MinMaxScaler()), ('xgb', XGBRegressor(base_score=None, booster=None, callbacks=None,
             colsample_bylevel=None, colsample_bynode=None,
             colsample_bytree=None, device=None, early_stopping_rounds=None,
             enable_categorical=False, eval_metric=None, feature_types=None,
             gamma=None, grow_policy=None, importance_type=None,
             interaction_constraints=None, learning_rate=None, max_bin=None,
             max_cat_threshold=None, max_cat_to_onehot=None,
             max_delta_step=None, max_depth=None, max_leaves=None,
             min_child_weight=None, missing=nan, monotone_constraints=None,
             multi_strategy=None, n_estimators=None, n_jobs=None,
             num_parallel_tree=None, random_state=None, ...))], 'verbose': False, 'scaler': MinMaxScaler(), 'xgb': XGBRegressor(base_score=None, booster=None, callbacks=None,
             colsample_bylevel=None, colsample_bynode=None,
             colsample_bytree=None, device=None, early_stopping_rounds=None,
             enable_categorical=False, eval_metric=None, feature_types=None,
             gamma=None, grow_policy=None, importance_type=None,
             interaction_constraints=None, learning_rate=None, max_bin=None,
             max_cat_threshold=None, max_cat_to_onehot=None,
             max_delta_step=None, max_depth=None, max_leaves=None,
             min_child_weight=None, missing=nan, monotone_constraints=None,
             multi_strategy=None, n_estimators=None, n_jobs=None,
             num_parallel_tree=None, random_state=None, ...), 'scaler__clip': False, 'scaler__copy': True, 'scaler__feature_range': (0, 1), 'xgb__objective': 'reg:squarederror', 'xgb__base_score': None, 'xgb__booster': None, 'xgb__callbacks': None, 'xgb__colsample_bylevel': None, 'xgb__colsample_bynode': None, 'xgb__colsample_bytree': None, 'xgb__device': None, 'xgb__early_stopping_rounds': None, 'xgb__enable_categorical': False, 'xgb__eval_metric': None, 'xgb__feature_types': None, 'xgb__gamma': None, 'xgb__grow_policy': None, 'xgb__importance_type': None, 'xgb__interaction_constraints': None, 'xgb__learning_rate': None, 'xgb__max_bin': None, 'xgb__max_cat_threshold': None, 'xgb__max_cat_to_onehot': None, 'xgb__max_delta_step': None, 'xgb__max_depth': None, 'xgb__max_leaves': None, 'xgb__min_child_weight': None, 'xgb__missing': nan, 'xgb__monotone_constraints': None, 'xgb__multi_strategy': None, 'xgb__n_estimators': None, 'xgb__n_jobs': None, 'xgb__num_parallel_tree': None, 'xgb__random_state': None, 'xgb__reg_alpha': None, 'xgb__reg_lambda': None, 'xgb__sampling_method': None, 'xgb__scale_pos_weight': None, 'xgb__subsample': None, 'xgb__tree_method': None, 'xgb__validate_parameters': None, 'xgb__verbosity': None}"""