# task3_modeling.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

# ---------- 1. Load Data ----------
def load_data(filepath):
    df = pd.read_csv(filepath, sep="|", parse_dates=['TransactionMonth'], low_memory=False)
    return df

# ---------- 2. Feature Engineering ----------
def feature_engineering(df):
    df = df.copy()

    # ✅ Ensure 'TransactionMonth' is datetime
    if 'TransactionMonth' in df.columns:
        df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'], errors='coerce')
    else:
        raise ValueError("🚨 'TransactionMonth' column is missing!")

    # Clean numeric columns
    numeric_cols = ['TotalPremium', 'TotalClaims', 'CustomValueEstimate']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Time features
    df['Month'] = df['TransactionMonth'].dt.month
    df['Year'] = df['TransactionMonth'].dt.year

    # Ratio features
    df['PremiumPerValue'] = df['TotalPremium'] / (df['CustomValueEstimate'] + 1)
    df['ClaimPerValue'] = df['TotalClaims'] / (df['CustomValueEstimate'] + 1)
    df['HasClaim'] = df['TotalClaims'].apply(lambda x: 1 if x > 0 else 0)

    # Drop rows with missing values (after conversion)
    df = df.dropna(subset=['TotalPremium', 'TotalClaims', 'CustomValueEstimate'])

    # Encode categoricals safely
    cat_cols = ['Province', 'Gender', 'VehicleType']
    for col in cat_cols:
        df[col] = df[col].astype(str).str.strip()

    # ✅ FIX: Keep all dummy columns (prevent SHAP/column mismatch issues)
    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=False)

    return df_encoded


# ---------- 3. Split Data ----------
def prepare_train_test(df_encoded):
    features = df_encoded.drop(columns=['TransactionMonth', 'TotalClaims'], errors='ignore')
    target = df_encoded['TotalClaims']

    # Ensure all features are numeric
    features = features.apply(pd.to_numeric, errors='coerce')
    features = features.fillna(0)

    return train_test_split(features, target, test_size=0.3, random_state=42)

# ---------- 4. Train Model ----------
def train_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# ---------- 5. Evaluate ----------
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5  # Compatible with all sklearn versions
    r2 = r2_score(y_test, y_pred)
    print(f"\n📊 RMSE: {rmse:.2f}")
    print(f"📈 R² Score: {r2:.2f}")
    return y_pred

# ---------- 6. Feature Importance ----------
def plot_feature_importance(model, X_train):
    feat_imp = pd.Series(model.feature_importances_, index=X_train.columns)
    top_feats = feat_imp.sort_values(ascending=False).head(10)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_feats.values, y=top_feats.index)
    plt.title("Top 10 Important Features")
    plt.xlabel("Feature Importance")
    plt.tight_layout()
    plt.show()

# ---------- 7. Save Model ----------
def save_model(model, filepath='../models/random_forest_model.pkl'):
    joblib.dump(model, filepath)
    print(f"\n✅ Model saved to: {filepath}")

# ---------- 8. Main ----------
if __name__ == "__main__":
    df = load_data("../data/MachineLearningRating_v3.txt")
    df_fe = feature_engineering(df)
    X_train, X_test, y_train, y_test = prepare_train_test(df_fe)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    plot_feature_importance(model, X_train)
    save_model(model)
