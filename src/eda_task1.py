# eda_task1.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

sns.set(style="whitegrid", palette="viridis")
plt.rcParams["figure.figsize"] = (12, 6)

# ---------- Data Summary ----------
def summarize_data(df):
    print("Data Info:\n", df.info())
    print("\nDescriptive Statistics:\n", df.describe(include='all'))
    print("\nColumn Types:\n", df.dtypes)

# ---------- Missing Values ----------
def check_missing_values(df):
    missing = df.isnull().sum()
    print("\nMissing Values:\n", missing[missing > 0])

# ---------- Loss Ratio ----------
def calculate_loss_ratio(df):
    df['LossRatio'] = df['TotalClaims'] / df['TotalPremium']
    return {
        'df': df,
        'by_province': df.groupby('Province')['LossRatio'].mean(),
        'by_gender': df.groupby('Gender')['LossRatio'].mean(),
        'by_vehicle': df.groupby('VehicleType')['LossRatio'].mean(),
    }

# ---------- Distributions ----------
def plot_distributions(df, numeric_cols, cat_cols):
    for col in numeric_cols:
        sns.histplot(df[col], kde=True)
        plt.title(f"Distribution of {col}")
        plt.show()

    for col in cat_cols:
        df[col].value_counts().plot(kind='bar')
        plt.title(f"Count of {col}")
        plt.show()

# ---------- Correlations ----------
def correlation_analysis(df, cols):
    corr = df[cols].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title("Correlation Matrix")
    plt.show()

# ---------- Bivariate ----------
def plot_scatter(df, x, y, hue):
    sns.lmplot(data=df, x=x, y=y, hue=hue, aspect=2, height=6)
    plt.title(f"{x} vs {y} by {hue}")
    plt.show()

def plot_temporal_trends(df, date_col):
    monthly_trends = df.groupby(df[date_col].dt.to_period("M"))[['TotalClaims', 'TotalPremium']].sum()
    monthly_trends.index = monthly_trends.index.to_timestamp()
    monthly_trends.plot(marker='o')
    plt.title("Monthly Total Premium vs Total Claim")
    plt.ylabel("Amount")
    plt.grid(True)
    plt.show()

# ---------- Top Claims ----------
def top_vehicle_claims(df):
    top_makes = df.groupby("make")["TotalClaims"].mean().sort_values(ascending=False).head(10)
    sns.barplot(x=top_makes.values, y=top_makes.index, palette="viridis")
    plt.title("Top 10 Vehicle Makes by Avg Claim Amount")
    plt.xlabel("Average Claim")
    plt.show()

# ---------- Outliers ----------
def detect_outliers(df, cols):
    for col in cols:
        sns.boxplot(x=df[col])
        plt.title(f"Boxplot of {col}")
        plt.show()

# ---------- Creative Plots ----------
def creative_plot_loss_ratio_by_province(df):
    fig = px.treemap(df, path=['Province'], values='LossRatio',
                     color='LossRatio', color_continuous_scale='RdBu',
                     title="Loss Ratio Distribution by Province")
    fig.show()

def creative_plot_claims_vs_premium(df):
    df_plot = df.dropna(subset=['TotalPremium', 'TotalClaims', 'make', 'CustomValueEstimate'])
    fig2 = px.scatter(df_plot, x='TotalPremium', y='TotalClaims', color='make',
                      size='CustomValueEstimate', title="Claim vs Premium by Vehicle Make")
    fig2.show()

def creative_plot_heatmap_loss_ratio(df):
    pivot = df.pivot_table(index='Gender', columns='VehicleType', values='LossRatio', aggfunc='mean')
    sns.heatmap(pivot, annot=True, cmap="YlGnBu")
    plt.title("Avg Loss Ratio by Gender and VehicleType")
    plt.show()
