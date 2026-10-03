import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned dataset
file_path = "Dataset/telco_churn_cleaned.csv"
df = pd.read_csv(file_path)

# Churn count by Internet Service
internet_churn = pd.crosstab(
    df["InternetService"],
    df["Churn"]
)

print("========== CHURN BY INTERNET SERVICE ==========\n")
print(internet_churn)

# Churn percentage within each Internet Service type
internet_churn_percentage = pd.crosstab(
    df["InternetService"],
    df["Churn"],
    normalize="index"
) * 100

print("\n========== CHURN PERCENTAGE BY INTERNET SERVICE ==========\n")
print(internet_churn_percentage)

# Visualization
plt.figure(figsize=(9, 6))

sns.countplot(
    data=df,
    x="InternetService",
    hue="Churn"
)

plt.title("Customer Churn by Internet Service")
plt.xlabel("Internet Service")
plt.ylabel("Number of Customers")
plt.legend(title="Churn")

plt.tight_layout()

plt.savefig(
    "Visualizations/churn_by_internet_service.png",
    dpi=300
)

plt.show()

print("\nInternet service churn graph saved successfully!")