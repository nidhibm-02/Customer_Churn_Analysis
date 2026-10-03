import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned dataset
file_path = "Dataset/telco_churn_cleaned.csv"
df = pd.read_csv(file_path)

# --------------------------------------------------
# 1. Basic statistical summary
# --------------------------------------------------

print("========== BASIC STATISTICS ==========\n")

print(df.describe())

# --------------------------------------------------
# 2. Churn count
# --------------------------------------------------

print("\n========== CHURN COUNT ==========\n")

print(df["Churn"].value_counts())

# --------------------------------------------------
# 3. Churn percentage
# --------------------------------------------------

print("\n========== CHURN PERCENTAGE ==========\n")

churn_percentage = df["Churn"].value_counts(normalize=True) * 100
print(churn_percentage)

# --------------------------------------------------
# 4. Churn distribution visualization
# --------------------------------------------------

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="Churn"
)

plt.title("Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")

plt.tight_layout()

# Save visualization
plt.savefig("Visualizations/churn_distribution.png", dpi=300)

plt.show()

print("\nChurn distribution graph saved successfully!")