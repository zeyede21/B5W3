# eda_utils.py or eda_task1.ipynb

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

sns.set(style="whitegrid", palette="viridis")
plt.rcParams["figure.figsize"] = (12, 6)

# Load your dataset
df = pd.read_csv("../data/MachineLearningRating_v3.txt", sep="|") # adjust filename
# Fix date format
df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'], errors='coerce')

# ---------------------s
# Data Summary
# ---------------------
print("Data Info:\n", df.info())
print("\nDescriptive Statistics:\n", df.describe(include='all'))

# Check column types
print("\nColumn Types:\n", df.dtypes)

# ---------------------
# Data Quality
# ---------------------
missing = df.isnull().sum()
print("\nMissing Values:\n", missing[missing > 0])

# ---------------------
# Loss Ratio Calculation
# ---------------------
df['LossRatio'] = df['TotalClaims'] / df['TotalPremium']
loss_by_province = df.groupby('Province')['LossRatio'].mean().sort_values(ascending=False)
loss_by_gender = df.groupby('Gender')['LossRatio'].mean()
loss_by_vehicle = df.groupby('VehicleType')['LossRatio'].mean()

# ---------------------
# Univariate Analysis
# ---------------------
num_cols = ['TotalPremium', 'TotalClaims', 'CustomValueEstimate']
cat_cols = ['Province', 'Gender', 'VehicleType']

for col in num_cols:
    sns.histplot(df[col], kde=True)
    plt.title(f"Distribution of {col}")
    plt.show()

for col in cat_cols:
    df[col].value_counts().plot(kind='bar')
    plt.title(f"Count of {col}")
    plt.show()

# ---------------------
# Bivariate Analysis
# ---------------------
# Correlation Matrix
corr = df[['TotalPremium', 'TotalClaims', 'CustomValueEstimate']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# Scatterplot with regression
sns.lmplot(data=df, x='TotalPremium', y='TotalClaims', hue='Province', aspect=2, height=6)
plt.title("TotalPremium vs TotalClaims by Province")
plt.show()

# Grouped by ZipCode (if available)
if 'ZipCode' in df.columns:
    zipcode_corr = df.groupby('ZipCode')[['TotalPremium', 'TotalClaims']].mean().corr()
    print("ZipCode Grouped Correlation:\n", zipcode_corr)

# ---------------------
# Temporal Trends
# ---------------------
monthly_trends = df.groupby(df['TransactionMonth'].dt.to_period("M"))[['TotalClaims', 'TotalPremium']].sum()
monthly_trends.index = monthly_trends.index.to_timestamp()

monthly_trends.plot(marker='o')
plt.title("Monthly Total Premium vs Total Claim")
plt.ylabel("Amount")
plt.grid(True)
plt.show()

# ---------------------
# Top Makes with Highest and Lowest Claims
# ---------------------
top_makes = df.groupby("make")["TotalClaims"].mean().sort_values(ascending=False).head(10)
sns.barplot(x=top_makes.values, y=top_makes.index, palette="viridis")
plt.title("Top 10 Vehicle Makes by Avg Claim Amount")
plt.xlabel("Average Claim")
plt.show()

# ---------------------
# Outlier Detection
# ---------------------
for col in num_cols:
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.show()

# ---------------------
# Creative Visuals
# ---------------------

# 1. Loss Ratio by Province (Plotly Treemap)
fig = px.treemap(df, path=['Province'], values='LossRatio',
                 color='LossRatio', color_continuous_scale='RdBu',
                 title="Loss Ratio Distribution by Province")
fig.show()

# 2. Premium vs Claim with Make as Color (Plotly Bubble Plot)
# Drop rows with NaN in columns required for the plot
df_plot = df.dropna(subset=['TotalPremium', 'TotalClaims', 'make', 'CustomValueEstimate'])

fig2 = px.scatter(df_plot, x='TotalPremium', y='TotalClaims', color='make',
                  size='CustomValueEstimate', title="Claim vs Premium by Vehicle Make")
fig2.show()


# 3. Heatmap of LossRatio by Gender and VehicleType
pivot = df.pivot_table(index='Gender', columns='VehicleType', values='LossRatio', aggfunc='mean')
sns.heatmap(pivot, annot=True, cmap="YlGnBu")
plt.title("Avg Loss Ratio by Gender and VehicleType")
plt.show()
