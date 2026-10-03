import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned dataset
file_path = "Dataset/telco_churn_cleaned.csv"
df = pd.read_csv(file_path)

# Churn count by payment method
payment_churn = pd.crosstab(
    df["PaymentMethod"],
    df["Churn"]
)

print("========== CHURN BY PAYMENT METHOD ==========\n")
print(payment_churn)

# Churn percentage within each payment method
payment_churn_percentage = pd.crosstab(
    df["PaymentMethod"],
    df["Churn"],
    normalize="index"
) * 100

print("\n========== CHURN PERCENTAGE BY PAYMENT METHOD ==========\n")
print(payment_churn_percentage)

# Visualization
plt.figure(figsize=(11, 6))

sns.countplot(
    data=df,
    x="PaymentMethod",
    hue="Churn"
)

plt.title("Customer Churn by Payment Method")
plt.xlabel("Payment Method")
plt.ylabel("Number of Customers")
plt.xticks(rotation=15)
plt.legend(title="Churn")

plt.tight_layout()

plt.savefig(
    "Visualizations/churn_by_payment_method.png",
    dpi=300
)

plt.show()

print("\nPayment method churn graph saved successfully!")