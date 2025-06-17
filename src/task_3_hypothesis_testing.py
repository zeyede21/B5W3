import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, f_oneway

# Load data
file_path = "../data/MachineLearningRating_v3.txt"
df = pd.read_csv(file_path, delimiter="|", low_memory=False)

# Clean and filter necessary columns
required_cols = ["TotalPremium", "TotalClaims", "Province", "PostalCode", "Gender"]
df = df.dropna(subset=required_cols)

# Fix column names if necessary
df.rename(columns={"PostalCode": "ZipCode"}, inplace=True)

# Compute Claim Frequency (at least one claim made)
df["ClaimFrequency"] = np.where(df["TotalClaims"] > 0, 1, 0)

# Compute Claim Severity (average claim amount where a claim occurred)
df["ClaimSeverity"] = df["TotalClaims"] / df["ClaimFrequency"].replace(0, np.nan)

# Compute Margin (TotalPremium - TotalClaims)
df["Margin"] = df["TotalPremium"] - df["TotalClaims"]

# ========== HYPOTHESIS 1: Risk differences across Provinces ==========
print("✅ H₀: No risk differences across Provinces (Claim Frequency)")
province_groups = [group["ClaimFrequency"] for _, group in df.groupby("Province")]
f_stat, p_value = f_oneway(*province_groups)
print(f"ANOVA result: F-stat = {f_stat:.4f}, p = {p_value:.4f}")
print("→ Reject H₀" if p_value < 0.05 else "→ Fail to reject H₀", end="\n\n")

# ========== HYPOTHESIS 2: Risk differences between Zip Codes ==========
print("✅ H₀: No risk differences between ZipCodes (Claim Frequency)")
zip_groups = [group["ClaimFrequency"] for _, group in df.groupby("ZipCode") if len(group) >= 30]
f_stat, p_value = f_oneway(*zip_groups)
print(f"ANOVA result: F-stat = {f_stat:.4f}, p = {p_value:.4f}")
print("→ Reject H₀" if p_value < 0.05 else "→ Fail to reject H₀", end="\n\n")

# ========== HYPOTHESIS 3: Margin difference between Zip Codes ==========
print("✅ H₀: No margin difference between ZipCodes")
zip_margin_groups = [group["Margin"] for _, group in df.groupby("ZipCode") if len(group) >= 30]
f_stat, p_value = f_oneway(*zip_margin_groups)
print(f"ANOVA result: F-stat = {f_stat:.4f}, p = {p_value:.4f}")
print("→ Reject H₀" if p_value < 0.05 else "→ Fail to reject H₀", end="\n\n")

# ========== HYPOTHESIS 4: Risk difference between Women and Men ==========
print("✅ H₀: No risk difference between Women and Men (Claim Frequency)")
female = df[df["Gender"] == "Female"]["ClaimFrequency"]
male = df[df["Gender"] == "Male"]["ClaimFrequency"]
t_stat, p_value = ttest_ind(female, male, equal_var=False)
print(f"T-test result: t = {t_stat:.4f}, p = {p_value:.4f}")
print("→ Reject H₀" if p_value < 0.05 else "→ Fail to reject H₀")
