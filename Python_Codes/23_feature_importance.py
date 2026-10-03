import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

print("========== FEATURE IMPORTANCE ANALYSIS ==========\n")

# Load cleaned dataset
file_path = "Dataset/telco_churn_cleaned.csv"
df = pd.read_csv(file_path)

# Remove customer ID
df = df.drop("customerID", axis=1)

# Separate features and target
X = df.drop("Churn", axis=1)
y = df["Churn"].map({
    "No": 0,
    "Yes": 1
})

# One-hot encoding
X = pd.get_dummies(X, drop_first=True, dtype=int)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
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

# Feature importance
importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": model.feature_importances_
})

# Sort by importance
importance = importance.sort_values(
    "Importance",
    ascending=False
)

print("----- Top 15 Important Features -----")
print(
    importance.head(15).to_string(index=False)
)

# Save feature importance
output_file = "Results/feature_importance.csv"
importance.to_csv(output_file, index=False)

print(f"\nFeature importance saved to: {output_file}")

# Plot top 15 features
top_features = importance.head(15).sort_values(
    "Importance"
)

plt.figure(figsize=(10, 7))

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.title("Top 15 Features Influencing Churn Prediction")
plt.xlabel("Feature Importance")
plt.ylabel("Feature")

plt.tight_layout()

chart_file = "Visualizations/feature_importance.png"
plt.savefig(chart_file, dpi=300)
plt.close()

print(f"Feature importance chart saved to: {chart_file}")

print("\n========== FEATURE IMPORTANCE ANALYSIS COMPLETED ==========")