import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load cleaned dataset
file_path = "Dataset/telco_churn_cleaned.csv"
df = pd.read_csv(file_path)

print("========== MACHINE LEARNING DATA PREPARATION ==========\n")

# Remove customer ID
df = df.drop("customerID", axis=1)

# Separate features and target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# One-hot encode categorical variables
X_encoded = pd.get_dummies(X, drop_first=True, dtype=int)

# Convert target into numerical values
y_encoded = y.map({
    "No": 0,
    "Yes": 1
})

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)

# Feature scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Original feature shape:", X.shape)
print("Encoded feature shape:", X_encoded.shape)

print("\nTraining set:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("\nTesting set:")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

print("\nScaled training data shape:", X_train_scaled.shape)
print("Scaled testing data shape:", X_test_scaled.shape)

print("\nTraining target distribution:")
print(y_train.value_counts())
print(y_train.value_counts(normalize=True) * 100)

print("\nTesting target distribution:")
print(y_test.value_counts())
print(y_test.value_counts(normalize=True) * 100)

print("\nFeature scaling completed successfully!")