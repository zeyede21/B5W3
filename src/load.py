import pandas as pd
import os

try:
    # Load raw data
    df = pd.read_csv("../data/MachineLearningRating_v3.txt", delimiter=',', on_bad_lines='skip')  # Adjust delimiter as needed

    # Strip whitespace and unify case
    df = df.apply(lambda col: col.str.strip().str.lower() if col.dtype == "object" else col)

    # Handle missing values
    df = df.dropna()

    # Drop irrelevant or duplicate columns
    df = df.drop(columns=["PolicyNumber"], errors="ignore")

    # Drop duplicates
    df = df.drop_duplicates()

    # Save cleaned data
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/df_cleaned.csv", index=False)
    print("✅ Cleaned data saved to data/processed/df_cleaned.csv")

except Exception as e:
    print(f"❌ An error occurred: {e}")