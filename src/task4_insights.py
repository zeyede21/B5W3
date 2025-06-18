# task4_insights.py

import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error, r2_score, accuracy_score,
    precision_score, recall_score, f1_score
)
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier

from task3_modeling import feature_engineering  # Ensure updated version is used

sns.set(style="whitegrid")
shap.initjs()

# 1. Load cleaned data
def load_cleaned_data(filepath="../data/df_cleaned.csv"):
    df = pd.read_csv(filepath, sep="|", low_memory=False)

    # ✅ Ensure TransactionMonth is converted to datetime
    if 'TransactionMonth' in df.columns:
        df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'], errors='coerce')
        print("✅ Converted 'TransactionMonth' to datetime")
    else:
        raise ValueError("🚨 'TransactionMonth' column is missing!")

    # ✅ Optional: Add HasClaim column if missing
    if 'HasClaim' not in df.columns:
        df['HasClaim'] = (df['TotalClaims'].fillna(0) > 0).astype(int)
        print("✅ Created 'HasClaim' column")

    return feature_engineering(df)


# 2. Train regression models for claim severity (TotalClaims > 0)
def train_regression_models(df):
    df_reg = df[df['TotalClaims'] > 0].copy()
    X = df_reg.drop(columns=['TransactionMonth', 'TotalClaims'], errors='ignore')
    y = df_reg['TotalClaims']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=100, random_state=42)
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        r2 = r2_score(y_test, y_pred)
        results[name] = {"model": model, "rmse": rmse, "r2": r2}
        print(f"✅ {name} - RMSE: {rmse:.2f}, R²: {r2:.2f}")

    return results, X_train

# 3. Binary classification for claim probability (HasClaim)
def train_classification_models(df):
    X = df.drop(columns=['TransactionMonth', 'TotalClaims', 'HasClaim'], errors='ignore')
    y = df['HasClaim']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss', random_state=42)
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        results[name] = {"model": model, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1}
        print(f"✅ {name} - Accuracy: {acc:.2f}, Precision: {prec:.2f}, Recall: {rec:.2f}, F1: {f1:.2f}")

    return results, X_train

# 4. SHAP Analysis
def run_shap_analysis(model, X_train, max_display=10):
    explainer = shap.Explainer(model)
    shap_values = explainer(X_train[:200])  # Limit for performance
    shap.plots.beeswarm(shap_values, max_display=max_display)
    shap.summary_plot(shap_values, X_train, plot_type="bar", max_display=max_display)

# ---------- Execution (Jupyter or Script) ----------
if __name__ == "__main__":
    print("🚀 Starting Task 4: Insight Generation and Reporting")

    df = load_cleaned_data()

    print("\n🔍 Running Severity Prediction Models...")
    reg_results, reg_X_train = train_regression_models(df)

    print("\n🔍 Running Claim Probability Models...")
    clf_results, clf_X_train = train_classification_models(df)

    print("\n🎯 SHAP Interpretability for Best Regressor (e.g., XGBoost)")
    best_reg_model = reg_results["XGBoost"]["model"]
    run_shap_analysis(best_reg_model, reg_X_train)

    print("\n🎯 SHAP Interpretability for Best Classifier (e.g., XGBoost)")
    best_clf_model = clf_results["XGBoost"]["model"]
    run_shap_analysis(best_clf_model, clf_X_train)

    print("\n✅ Task 4 Completed")
