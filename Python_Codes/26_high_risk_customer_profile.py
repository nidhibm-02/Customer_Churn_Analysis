import pandas as pd

print("========== HIGH-RISK CUSTOMER PROFILE ANALYSIS ==========\n")

# ==================================================
# LOAD DATA
# ==================================================

customer_file = "Dataset/telco_churn_cleaned.csv"
risk_file = "Results/customer_churn_risk_analysis.csv"

customers = pd.read_csv(customer_file)
risk = pd.read_csv(risk_file)


# ==================================================
# MERGE CUSTOMER DATA WITH RISK RESULTS
# ==================================================

profile = pd.merge(
    customers,
    risk[[
        "customerID",
        "Churn_Probability",
        "Risk_Level"
    ]],
    on="customerID",
    how="inner"
)

print("Customers included in risk analysis:", len(profile))


# ==================================================
# HIGH-RISK CUSTOMERS
# ==================================================

high_risk = profile[
    profile["Risk_Level"] == "High Risk"
].copy()

print(
    "\nNumber of high-risk customers:",
    len(high_risk)
)

print(
    "Percentage of high-risk customers:",
    round(
        len(high_risk) / len(profile) * 100,
        2
    ),
    "%"
)


# ==================================================
# HIGH-RISK CONTRACT PROFILE
# ==================================================

print("\n----- High-Risk Customers by Contract -----")

contract_profile = (
    high_risk["Contract"]
    .value_counts()
    .reset_index()
)

contract_profile.columns = [
    "Contract",
    "High_Risk_Customers"
]

print(
    contract_profile.to_string(index=False)
)


# ==================================================
# HIGH-RISK INTERNET SERVICE PROFILE
# ==================================================

print("\n----- High-Risk Customers by Internet Service -----")

internet_profile = (
    high_risk["InternetService"]
    .value_counts()
    .reset_index()
)

internet_profile.columns = [
    "InternetService",
    "High_Risk_Customers"
]

print(
    internet_profile.to_string(index=False)
)


# ==================================================
# HIGH-RISK PAYMENT PROFILE
# ==================================================

print("\n----- High-Risk Customers by Payment Method -----")

payment_profile = (
    high_risk["PaymentMethod"]
    .value_counts()
    .reset_index()
)

payment_profile.columns = [
    "PaymentMethod",
    "High_Risk_Customers"
]

print(
    payment_profile.to_string(index=False)
)


# ==================================================
# HIGH-RISK TENURE PROFILE
# ==================================================

print("\n----- High-Risk Customers by Tenure -----")

print(
    high_risk["tenure"].describe()
)


# ==================================================
# HIGH-RISK MONTHLY CHARGES
# ==================================================

print("\n----- High-Risk Customers by Monthly Charges -----")

print(
    high_risk["MonthlyCharges"].describe()
)


# ==================================================
# SAVE HIGH-RISK PROFILE
# ==================================================

high_risk.to_csv(
    "Results/high_risk_customer_profile.csv",
    index=False
)

print(
    "\nHigh-risk customer profile saved successfully."
)

print(
    "\nOutput file: Results/high_risk_customer_profile.csv"
)

print(
    "\n========== HIGH-RISK PROFILE ANALYSIS COMPLETED =========="
)