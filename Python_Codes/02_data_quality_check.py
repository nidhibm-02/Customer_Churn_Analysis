import pandas as pd

# Load the dataset
file_path = "Dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv"
df = pd.read_csv(file_path)

print("========== DATA QUALITY CHECK ==========\n")

# 1. Dataset shape
print("1. Dataset Shape:")
print(df.shape)

# 2. Missing values
print("\n2. Missing Values:")
print(df.isnull().sum())

# 3. Duplicate rows
print("\n3. Duplicate Rows:")
print(df.duplicated().sum())

# 4. Duplicate customer IDs
print("\n4. Duplicate Customer IDs:")
print(df["customerID"].duplicated().sum())

# 5. Unique values in important columns
print("\n5. Unique Values:")

columns_to_check = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "InternetService",
    "Contract",
    "PaymentMethod",
    "Churn"
]

for column in columns_to_check:
    print(f"\n{column}:")
    print(df[column].unique())

# 6. Churn distribution
print("\n6. Churn Distribution:")
print(df["Churn"].value_counts())

# 7. Churn percentage
print("\n7. Churn Percentage:")
print(df["Churn"].value_counts(normalize=True) * 100)

# 8. TotalCharges data type
print("\n8. TotalCharges Data Type:")
print(df["TotalCharges"].dtype)