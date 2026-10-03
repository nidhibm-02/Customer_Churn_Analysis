import pandas as pd
from scipy.stats import chi2_contingency
import numpy as np

# Load cleaned dataset
file_path = "Dataset/telco_churn_cleaned.csv"
df = pd.read_csv(file_path)

# Extended categorical variables
categorical_features = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "PaperlessBilling"
]

print("========== EXTENDED CHI-SQUARE ANALYSIS ==========\n")

results = []

for feature in categorical_features:

    # Contingency table
    table = pd.crosstab(df[feature], df["Churn"])

    # Chi-square test
    chi2, p, dof, expected = chi2_contingency(table)

    # Cramer's V
    n = table.to_numpy().sum()
    phi2 = chi2 / n
    rows, columns = table.shape

    cramer_v = np.sqrt(
        phi2 / min(columns - 1, rows - 1)
    )

    # Store results
    results.append({
        "Feature": feature,
        "Chi_Square": chi2,
        "Degrees_of_Freedom": dof,
        "P_Value": p,
        "Cramers_V": cramer_v
    })

    print(f"--- {feature} vs Churn ---")
    print(f"Chi-square statistic: {chi2:.4f}")
    print(f"Degrees of freedom: {dof}")
    print(f"p-value: {p:.6f}")
    print(f"Cramer's V: {cramer_v:.4f}")

    if p < 0.05:
        print("Result: Statistically significant association with churn.")
    else:
        print("Result: No statistically significant association with churn.")

    print("\n" + "=" * 60 + "\n")


# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Save statistical results
output_file = "Results/extended_chi_square_results.csv"
results_df.to_csv(output_file, index=False)

print("Extended chi-square results saved successfully!")
print(f"File: {output_file}")