import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned dataset
file_path = "Dataset/telco_churn_cleaned.csv"
df = pd.read_csv(file_path)

# Basic statistics of Monthly Charges by Churn
monthly_charges_stats = df.groupby("Churn")["MonthlyCharges"].describe()

print("========== MONTHLY CHARGES BY CHURN ==========\n")
print(monthly_charges_stats)

# Visualization
plt.figure(figsize=(9, 6))

sns.boxplot(
    data=df,
    x="Churn",
    y="MonthlyCharges"
)

plt.title("Monthly Charges Distribution by Churn")
plt.xlabel("Churn")
plt.ylabel("Monthly Charges")

plt.tight_layout()

plt.savefig(
    "Visualizations/monthly_charges_vs_churn.png",
    dpi=300
)

plt.show()

print("\nMonthly charges churn graph saved successfully!")