import pandas as pd

print("========== BUSINESS & RETENTION INSIGHTS ==========\n")

# Load cleaned dataset
file_path = "Dataset/telco_churn_cleaned.csv"
df = pd.read_csv(file_path)

# Convert Churn to numeric
df["Churn_Flag"] = df["Churn"].map({
    "No": 0,
    "Yes": 1
})

# Function to generate segment analysis
def analyze_segment(data, column_name):
    result = (
        data.groupby(column_name)
        .agg(
            Customers=("Churn_Flag", "count"),
            Churned=("Churn_Flag", "sum"),
            Churn_Rate=("Churn_Flag", "mean")
        )
        .reset_index()
    )

    result["Churn_Rate"] = (
        result["Churn_Rate"] * 100
    ).round(2)

    return result


# ==================================================
# 1. CONTRACT TYPE
# ==================================================

print("----- Contract Type Analysis -----")

contract_analysis = analyze_segment(
    df,
    "Contract"
)

print(contract_analysis.to_string(index=False))


# ==================================================
# 2. INTERNET SERVICE
# ==================================================

print("\n----- Internet Service Analysis -----")

internet_analysis = analyze_segment(
    df,
    "InternetService"
)

print(internet_analysis.to_string(index=False))


# ==================================================
# 3. PAYMENT METHOD
# ==================================================

print("\n----- Payment Method Analysis -----")

payment_analysis = analyze_segment(
    df,
    "PaymentMethod"
)

print(payment_analysis.to_string(index=False))


# ==================================================
# SAVE RESULTS
# ==================================================

contract_analysis.to_csv(
    "Results/contract_business_insights.csv",
    index=False
)

internet_analysis.to_csv(
    "Results/internet_service_business_insights.csv",
    index=False
)

payment_analysis.to_csv(
    "Results/payment_method_business_insights.csv",
    index=False
)

print("\nBusiness insight files saved successfully.")

print("\n========== BUSINESS & RETENTION ANALYSIS COMPLETED ==========")