import pandas as pd

print("========== CUSTOMER CHURN PROJECT SUMMARY ==========\n")

# ==================================================
# LOAD DATA
# ==================================================

df = pd.read_csv(
    "Dataset/telco_churn_cleaned.csv"
)

risk = pd.read_csv(
    "Results/customer_churn_risk_analysis.csv"
)

models = pd.read_csv(
    "Results/final_model_performance.csv"
)


# ==================================================
# DATASET SUMMARY
# ==================================================

total_customers = len(df)

total_churned = (
    df["Churn"] == "Yes"
).sum()

overall_churn_rate = (
    total_churned / total_customers
) * 100


# ==================================================
# HIGH-RISK SUMMARY
# ==================================================

high_risk_count = (
    risk["Risk_Level"] == "High Risk"
).sum()

high_risk_percentage = (
    high_risk_count / len(risk)
) * 100


# ==================================================
# KEY BUSINESS SEGMENTS
# ==================================================

contract_churn = (
    df.groupby("Contract")["Churn"]
    .apply(lambda x: (x == "Yes").mean() * 100)
    .round(2)
)

internet_churn = (
    df.groupby("InternetService")["Churn"]
    .apply(lambda x: (x == "Yes").mean() * 100)
    .round(2)
)

payment_churn = (
    df.groupby("PaymentMethod")["Churn"]
    .apply(lambda x: (x == "Yes").mean() * 100)
    .round(2)
)


# ==================================================
# MODEL METRICS
# ==================================================

best_roc_auc = models.loc[
    models["ROC-AUC"].idxmax()
]

best_accuracy = models.loc[
    models["Accuracy"].idxmax()
]

best_recall = models.loc[
    models["Recall"].idxmax()
]


# ==================================================
# DISPLAY SUMMARY
# ==================================================

print("----- Dataset Summary -----")

print(
    "Total customers:",
    total_customers
)

print(
    "Total churned customers:",
    total_churned
)

print(
    "Overall churn rate:",
    round(overall_churn_rate, 2),
    "%"
)


print("\n----- High-Risk Customer Summary -----")

print(
    "High-risk customers:",
    high_risk_count
)

print(
    "High-risk percentage:",
    round(high_risk_percentage, 2),
    "%"
)


print("\n----- Contract Churn Rates -----")

print(contract_churn)


print("\n----- Internet Service Churn Rates -----")

print(internet_churn)


print("\n----- Payment Method Churn Rates -----")

print(payment_churn)


print("\n----- Model Summary -----")

print(
    "Highest ROC-AUC:",
    best_roc_auc["Model"],
    "=",
    best_roc_auc["ROC-AUC"]
)

print(
    "Highest Accuracy:",
    best_accuracy["Model"],
    "=",
    best_accuracy["Accuracy"]
)

print(
    "Highest Recall:",
    best_recall["Model"],
    "=",
    best_recall["Recall"]
)


# ==================================================
# SAVE SUMMARY
# ==================================================

summary = pd.DataFrame({
    "Metric": [
        "Total Customers",
        "Total Churned Customers",
        "Overall Churn Rate (%)",
        "High-Risk Customers",
        "High-Risk Customers (%)",
        "Highest ROC-AUC",
        "Highest Accuracy",
        "Highest Recall"
    ],

    "Value": [
        total_customers,
        total_churned,
        round(overall_churn_rate, 2),
        high_risk_count,
        round(high_risk_percentage, 2),
        best_roc_auc["Model"],
        best_accuracy["Model"],
        best_recall["Model"]
    ]
})


summary.to_csv(
    "Results/project_summary.csv",
    index=False
)


print(
    "\nProject summary saved successfully."
)

print(
    "Output file: Results/project_summary.csv"
)

print(
    "\n========== PROJECT SUMMARY COMPLETED =========="
)