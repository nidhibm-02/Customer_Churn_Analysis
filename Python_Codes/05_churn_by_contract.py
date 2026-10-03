import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned dataset
file_path = "Dataset/telco_churn_cleaned.csv"
df = pd.read_csv(file_path)

# --------------------------------------------------
# Churn count by contract type
# --------------------------------------------------

contract_churn = pd.crosstab(
    df["Contract"],
    df["Churn"]
)

print("========== CHURN BY CONTRACT TYPE ==========\n")
print(contract_churn)

# --------------------------------------------------
# Churn percentage within each contract type
# --------------------------------------------------

contract_churn_percentage = pd.crosstab(
    df["Contract"],
    df["Churn"],
    normalize="index"
) * 100

print("\n========== CHURN PERCENTAGE BY CONTRACT TYPE ==========\n")
print(contract_churn_percentage)

# --------------------------------------------------
# Visualization
# --------------------------------------------------

plt.figure(figsize=(9, 6))

sns.countplot(
    data=df,
    x="Contract",
    hue="Churn"
)

plt.title("Customer Churn by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Number of Customers")
plt.legend(title="Churn")

plt.tight_layout()

plt.savefig(
    "Visualizations/churn_by_contract.png",
    dpi=300
)

plt.show()

print("\nContract churn graph saved successfully!")