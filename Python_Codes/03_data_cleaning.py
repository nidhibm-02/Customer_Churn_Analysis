import pandas as pd

# Load original dataset
file_path = "Dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv"
df = pd.read_csv(file_path)

print("Original dataset shape:", df.shape)

# --------------------------------------------------
# 1. Convert TotalCharges to numeric
# --------------------------------------------------

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# Check missing values created during conversion
print("\nMissing TotalCharges after conversion:")
print(df["TotalCharges"].isnull().sum())

# --------------------------------------------------
# 2. Handle missing TotalCharges values
# --------------------------------------------------

# TotalCharges is related to tenure and monthly charges.
# For customers with zero tenure, TotalCharges should be 0.
df["TotalCharges"] = df["TotalCharges"].fillna(0)

# --------------------------------------------------
# 3. Check duplicates
# --------------------------------------------------

print("\nDuplicate rows:", df.duplicated().sum())
print("Duplicate customer IDs:", df["customerID"].duplicated().sum())

# --------------------------------------------------
# 4. Convert SeniorCitizen to meaningful labels
# --------------------------------------------------

df["SeniorCitizen"] = df["SeniorCitizen"].map({
    0: "No",
    1: "Yes"
})

# --------------------------------------------------
# 5. Save cleaned dataset
# --------------------------------------------------

output_file = "Dataset/telco_churn_cleaned.csv"

df.to_csv(output_file, index=False)

print("\nCleaned dataset shape:", df.shape)
print("\nCleaned data types:")
print(df.dtypes)

print("\nCleaned dataset saved successfully!")
print(output_file)