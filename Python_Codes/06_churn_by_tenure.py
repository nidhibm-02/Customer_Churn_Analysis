import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned dataset
file_path = "Dataset/telco_churn_cleaned.csv"
df = pd.read_csv(file_path)

# Create tenure groups
df["TenureGroup"] = pd.cut(
    df["tenure"],
    bins=[-1, 12, 24, 48, 72],
    labels=[
        "0-12 Months",
        "13-24 Months",
        "25-48 Months",
        "49-72 Months"
    ]
)

# Churn count by tenure group
tenure_churn = pd.crosstab(
    df["TenureGroup"],
    df["Churn"]
)

print("========== CHURN BY TENURE GROUP ==========\n")
print(tenure_churn)

# Churn percentage within each tenure group
tenure_churn_percentage = pd.crosstab(
    df["TenureGroup"],
    df["Churn"],
    normalize="index"
) * 100

print("\n========== CHURN PERCENTAGE BY TENURE GROUP ==========\n")
print(tenure_churn_percentage)

# Visualization
plt.figure(figsize=(9, 6))

sns.countplot(
    data=df,
    x="TenureGroup",
    hue="Churn"
)

plt.title("Customer Churn by Tenure Group")
plt.xlabel("Tenure Group")
plt.ylabel("Number of Customers")
plt.legend(title="Churn")

plt.tight_layout()

plt.savefig(
    "Visualizations/churn_by_tenure.png",
    dpi=300
)

plt.show()

print("\nTenure churn graph saved successfully!")