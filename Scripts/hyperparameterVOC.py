# VOC FOR HYPERPARAMETER OPTIMIZATION

# This code uses the framework of the VOC prediction.

import csv, math
import pandas as pd
import numpy as np
from numpy import nan_to_num, std
from sklearn import linear_model, ensemble, svm, neural_network
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RepeatedKFold, cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Algorithm to test
models = [RandomForestRegressor(random_state=47)]
data = pd.read_csv('VOC-Database.csv')

exclude_cols = ['CAS', 'dvap', 'num', 'External', 'SMILES', 'Key', 'Family', 'VOC']
X = data.drop(columns=exclude_cols)
y = data['dvap']

# Split data based on 'External' column
train_data = data[data['VOC'] == 'NO']
test_data = data[data['VOC'] == 'YES']

X_train = train_data.drop(columns=exclude_cols)
y_train = train_data['dvap'].to_numpy()
X_test = test_data.drop(columns=exclude_cols)
y_test = test_data['dvap'].to_numpy()

X_train.fillna(0, inplace=True)
X_test.fillna(0, inplace=True)

n_iterations = 20

results = []

for model in models:
    for iteration in range(n_iterations):                
        model.fit(X_train, y_train)

        # Calculate feature importance
        p_imp = permutation_importance(model, X_test, y_test, n_repeats=5)
        p_imp_av = p_imp['importances_mean']
        feature_importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': p_imp_av})


        relative_importance = (p_imp_av / sum(p_imp_av))
        bad_features = [name for name, importance in zip(X_train.columns, relative_importance) if importance < 0.00001]
        good_features = [name for name in X_train.columns if name not in bad_features]
        X_train_filtered = X_train[good_features]
        X_test_filtered = X_test[good_features]


        X_train_filtered.fillna(0, inplace=True)
        X_test_filtered.fillna(0, inplace=True)


        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Calculate metrics
        rsq_train = model.score(X_train, y_train)
        rsq_test = model.score(X_test, y_test)
        Score_train = np.corrcoef(y_train, y_train_pred)[0, 1] ** 2
        Score_test = np.corrcoef(y_test, y_test_pred)[0, 1] ** 2
        MSE = np.square(np.subtract(y_test, y_test_pred)).mean()
        RMSE = math.sqrt(MSE)
        cv = RepeatedKFold(n_splits=2, n_repeats=3, random_state=iteration)
        n_scores = cross_val_score(model, X_test, y_test, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')
        STD = std(n_scores)
        MAE = mean_absolute_error(y_test, y_test_pred)
        AARD = (100 / len(X_test_filtered)) * sum(abs((y_test_pred - y_test) / y_test_pred))
        
    
        
        # Record results
        result = {
            "Algorithm": model.__class__.__name__,
            "Iteration": iteration + 1,
            "rsq_train": rsq_train,
            "rsq_test": rsq_test,
            "Score_train": Score_train,
            "Score_test": Score_test,
            "RMSE": RMSE,
            "MAE": MAE,
            "STD": STD,
            "AARD": AARD,
            "N Bad Features": len(bad_features)}
        results.append(result)

# Output file
output_fn = 'VOC-results.csv'
with open(output_fn, 'w', newline='') as fout:
    writer = csv.DictWriter(fout, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)



### Pipeline
# Calcualte best hyperparameters

steps = [('scaler', StandardScaler()), ('Forest', RandomForestRegressor())]
pipeline = Pipeline(steps)

parameters = {'Forest__n_estimators': [100, 200, 300],
              'Forest__max_features': ['sqrt','log2'],
              'Forest__min_samples_split': [2, 5],
              'Forest__min_samples_leaf': [1, 2],
              'Forest__max_depth': [10, 20, 40],
              'Forest__bootstrap': [True, False],
              'Forest__warm_start': [True, False]}

model = RandomizedSearchCV(pipeline, parameters, n_iter=5, scoring='neg_mean_absolute_error', cv=10)

model.fit(X_train, y_train)
Best_Parameters = model.best_params_

best_params_df = pd.DataFrame([model.best_params_])
best_params_df["best_cv_score_neg_mae"] = model.best_score_
best_params_df.to_csv("best_hyperparameters.csv", index=False)

with open("best_hyperparameters.txt", "w") as f:
    f.write(f"Best CV score (neg MAE): {model.best_score_}\n")
    f.write("Best params:\n")
    for k, v in model.best_params_.items():
        f.write(f"{k}: {v}\n")
        
cv_results_df = pd.DataFrame(model.cv_results_)

# keep the most useful columns first
cols_first = [c for c in cv_results_df.columns if c.startswith("param_")]
cols_first += ["mean_test_score", "std_test_score", "rank_test_score", "mean_fit_time", "mean_score_time"]
cols_first = [c for c in cols_first if c in cv_results_df.columns]
cv_results_df = cv_results_df[cols_first + [c for c in cv_results_df.columns if c not in cols_first]]

cv_results_df.to_csv("random_search_cv_results.csv", index=False)
