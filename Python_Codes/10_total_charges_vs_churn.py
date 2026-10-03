import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned dataset
file_path = "Dataset/telco_churn_cleaned.csv"
df = pd.read_csv(file_path)

# Basic statistics of Total Charges by Churn
total_charges_stats = df.groupby("Churn")["TotalCharges"].describe()

print("========== TOTAL CHARGES BY CHURN ==========\n")
print(total_charges_stats)

# Visualization
plt.figure(figsize=(9, 6))

sns.boxplot(
    data=df,
    x="Churn",
    y="TotalCharges"
)

plt.title("Total Charges Distribution by Churn")
plt.xlabel("Churn")
plt.ylabel("Total Charges")

plt.tight_layout()

plt.savefig(
    "Visualizations/total_charges_vs_churn.png",
    dpi=300
)

plt.show()

print("\nTotal charges churn graph saved successfully!")