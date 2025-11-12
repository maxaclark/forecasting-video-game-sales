import pandas as pd
import re
import os

# Load dataset
DATA_PATH = "vgsales.csv"
OUTPUT_PATH = "vgsales_clean.csv"

# Data Frame
df = pd.read_csv(DATA_PATH)
print(f"1. Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

# Dealing with missing values
df = df.dropna(subset=["Year", "Global_Sales"])
df["Publisher"] = df["Publisher"].fillna("Unknown")

# Convert Year to int
df["Year"] = df["Year"].astype(int)
print(f"2. After cleaning nulls: {df.shape[0]} rows remain")

# Remove Duplicates
before = df.shape[0]
df = df.drop_duplicates(subset=["Name", "Platform"])
print(f"3. Removed {before - df.shape[0]} duplicate rows")

# Text Normalization
def clean_text(x):
    if pd.isnull(x):
        return ""
    x = x.lower().strip()
    x = re.sub(r"[^a-z0-9\s]", "", x)
    return x

for col in ["Name", "Platform", "Genre", "Publisher"]:
    df[col] = df[col].apply(clean_text)
print("4. Normalized text fields")

# Create Derived Features
df["Decade"] = (df["Year"] // 10) * 10
df["FranchiseTag"] = df["Name"].apply(
    lambda x: 1 if re.search(r"\b(ii|iii|iv|v|[2-9])\b", x) else 0
)
df["Total_Regional_Sales"] = df[["NA_Sales","EU_Sales","JP_Sales","Other_Sales"]].sum(axis=1)
df["Sales_Ratio"] = (df["Global_Sales"] / (df["Total_Regional_Sales"] + 1e-6)).round(2)

# Output format
df.to_csv(OUTPUT_PATH, index=False)

# Summary
print("\nSummary:")
print(df.head(5))
print("\nMissing values:\n", df.isna().sum())
print("\nData types:\n", df.dtypes)
