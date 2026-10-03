import pandas as pd

# Load the dataset
file_path = "Dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv"

df = pd.read_csv(file_path)

# Display first 5 rows
print("First 5 rows:")
print(df.head())

# Display dataset dimensions
print("\nDataset Shape:")
print(df.shape)

# Display column names
print("\nColumn Names:")
print(df.columns.tolist())

# Display data types
print("\nData Types:")
print(df.dtypes)

# Display basic information
print("\nDataset Information:")
df.info()