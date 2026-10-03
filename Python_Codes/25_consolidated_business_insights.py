import pandas as pd

print("========== CONSOLIDATED BUSINESS INSIGHTS ==========\n")

# Load cleaned dataset
file_path = "Dataset/telco_churn_cleaned.csv"
df = pd.read_csv(file_path)

# Convert Churn to numeric
df["Churn_Flag"] = df["Churn"].map({
    "No": 0,
    "Yes": 1
})


# ==================================================
# FUNCTION FOR SEGMENT ANALYSIS
# ==================================================

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
# CONTRACT ANALYSIS
# ==================================================

contract = analyze_segment(
    df,
    "Contract"
)

contract["Segment_Type"] = "Contract"


# ==================================================
# INTERNET SERVICE ANALYSIS
# ==================================================

internet = analyze_segment(
    df,
    "InternetService"
)

internet["Segment_Type"] = "Internet Service"


# ==================================================
# PAYMENT METHOD ANALYSIS
# ==================================================

payment = analyze_segment(
    df,
    "PaymentMethod"
)

payment["Segment_Type"] = "Payment Method"


# ==================================================
# STANDARDIZE COLUMN NAMES
# ==================================================

contract = contract.rename(
    columns={"Contract": "Segment"}
)

internet = internet.rename(
    columns={"InternetService": "Segment"}
)

payment = payment.rename(
    columns={"PaymentMethod": "Segment"}
)


# ==================================================
# COMBINE RESULTS
# ==================================================

consolidated = pd.concat(
    [
        contract,
        internet,
        payment
    ],
    ignore_index=True
)

consolidated = consolidated[
    [
        "Segment_Type",
        "Segment",
        "Customers",
        "Churned",
        "Churn_Rate"
    ]
]


# ==================================================
# DISPLAY RESULTS
# ==================================================

print(consolidated.to_string(index=False))


# ==================================================
# SAVE FINAL RESULTS
# ==================================================

output_file = (
    "Results/consolidated_business_insights.csv"
)

consolidated.to_csv(
    output_file,
    index=False
)

print(
    "\nConsolidated business insights saved successfully."
)

print(
    "\nOutput file:",
    output_file
)

print(
    "\n========== CONSOLIDATED BUSINESS ANALYSIS COMPLETED =========="
)