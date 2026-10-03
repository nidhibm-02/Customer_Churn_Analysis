import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

print("========== CUSTOMER CHURN RISK ANALYSIS ==========\n")

# Load cleaned dataset
file_path = "Dataset/telco_churn_cleaned.csv"
df = pd.read_csv(file_path)

# Keep customer IDs for final risk analysis
customer_ids = df["customerID"]

# Remove customer ID for ML
df_ml = df.drop("customerID", axis=1)

# Separate features and target
X = df_ml.drop("Churn", axis=1)
y = df_ml["Churn"].map({
    "No": 0,
    "Yes": 1
})

# One-hot encoding
X = pd.get_dummies(X, drop_first=True, dtype=int)

# Train-test split
X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
    X,
    y,
    customer_ids,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Tuned Gradient Boosting model
model = GradientBoostingClassifier(
    n_estimators=150,
    learning_rate=0.05,
    max_depth=2,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Predict churn probability
churn_probability = model.predict_proba(X_test)[:, 1]

# ROC-AUC
roc_auc = roc_auc_score(y_test, churn_probability)

print(f"Model ROC-AUC: {roc_auc:.4f}")

# Create risk dataframe
risk_df = pd.DataFrame({
    "customerID": id_test.values,
    "Actual_Churn": y_test.values,
    "Churn_Probability": churn_probability
})

# Assign risk levels
def assign_risk(probability):
    if probability < 0.30:
        return "Low Risk"
    elif probability < 0.60:
        return "Medium Risk"
    else:
        return "High Risk"

risk_df["Risk_Level"] = risk_df["Churn_Probability"].apply(assign_risk)

# Sort by highest churn probability
risk_df = risk_df.sort_values(
    "Churn_Probability",
    ascending=False
)

# Display risk distribution
print("\n----- Risk Level Distribution -----")
print(risk_df["Risk_Level"].value_counts())

print("\n----- Risk Level Percentage -----")
print(
    (risk_df["Risk_Level"].value_counts(normalize=True) * 100)
    .round(2)
)

# Display high-risk customers
print("\n----- Top 20 High-Risk Customers -----")
print(
    risk_df[
        [
            "customerID",
            "Actual_Churn",
            "Churn_Probability",
            "Risk_Level"
        ]
    ].head(20).to_string(index=False)
)

# Save complete risk analysis
output_file = "Results/customer_churn_risk_analysis.csv"
risk_df.to_csv(output_file, index=False)

print(f"\nRisk analysis saved to: {output_file}")

print("\n========== CHURN RISK ANALYSIS COMPLETED ==========")