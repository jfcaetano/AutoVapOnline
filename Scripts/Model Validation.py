#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 21 18:05:19 2025

@author: jfcaetano
"""

import csv, math
import pandas as pd
import numpy as np
from numpy import std
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RepeatedKFold, cross_val_score


desc_types = pd.read_csv("desc-types.csv")

models = [
    RandomForestRegressor(
        n_estimators=300,
        min_samples_split=2,
        min_samples_leaf=1,
        max_depth=20,
        random_state=47
    )
]

# Load data
data = pd.read_csv('Database-Global.csv')

exclude_cols = ['CAS', 'dvap', 'num', 'External', 'SMILES', 'Key', 'Family', 'VOC']
X = data.drop(columns=exclude_cols)
y = data['dvap']

n_iterations = 20
results = []  # to collect model performance metrics

all_feature_names = X.columns.tolist()
importance_history = []

for model in models:
    for iteration in range(n_iterations):

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.60, random_state=iteration
        )

        X_train = X_train.fillna(0)
        X_test  = X_test.fillna(0)

        # Fit model
        model.fit(X_train, y_train)

        p_imp = permutation_importance(
            model, X_test, y_test, n_repeats=1, random_state=iteration
        )
        p_imp_av = p_imp['importances_mean']

        # Normalize importances so they sum to 1
        total_imp = np.sum(p_imp_av)
        if total_imp == 0:
            rel_importances = np.zeros_like(p_imp_av)
        else:
            rel_importances = p_imp_av / total_imp

        importance_dict = dict(zip(all_feature_names, rel_importances))
        importance_history.append(importance_dict)

        # ---------- Metrics ----------
        y_train_pred = model.predict(X_train)
        y_test_pred  = model.predict(X_test)

        rsq_train = model.score(X_train, y_train)
        rsq_test  = model.score(X_test, y_test)

        Score_train = np.corrcoef(y_train, y_train_pred)[0, 1] ** 2
        Score_test  = np.corrcoef(y_test,  y_test_pred)[0, 1] ** 2

        MSE  = np.square(np.subtract(y_test, y_test_pred)).mean()
        RMSE = math.sqrt(MSE)

        cv = RepeatedKFold(n_splits=2, n_repeats=3, random_state=iteration)
        n_scores = cross_val_score(
            model, X_train, y_train,
            scoring='neg_mean_absolute_error',
            cv=cv, n_jobs=-1, error_score='raise'
        )
        STD = std(n_scores)

        MAE = mean_absolute_error(y_test, y_test_pred)

        denom = np.where(np.abs(y_test_pred) < 1e-12, 1e-12, y_test_pred)
        AARD = (100.0 / len(X_test)) * np.sum(np.abs((y_test_pred - y_test) / denom))

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
            "AARD": AARD
        }
        results.append(result)

# ------------------------------
output_fn = 'results.csv'
with open(output_fn, 'w', newline='') as fout:
    writer = csv.DictWriter(fout, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

imp_df = pd.DataFrame(importance_history)

avg_imp_df = pd.DataFrame({
    "Feature": imp_df.columns,
    "AvgImportance": imp_df.mean().values * 100.0  # scale to percentage
})

avg_imp_df = avg_imp_df.merge(desc_types, how="left", on="Feature")
avg_imp_df = avg_imp_df.sort_values(by="AvgImportance", ascending=False)

total_sum = avg_imp_df["AvgImportance"].sum()
# Save to CSV
avg_imp_df.to_csv("feature-importance.csv", index=False)
