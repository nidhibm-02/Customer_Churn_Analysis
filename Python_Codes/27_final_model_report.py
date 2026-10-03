import pandas as pd

print("========== FINAL MODEL PERFORMANCE REPORT ==========\n")

# ==================================================
# LOAD BASELINE MODEL RESULTS
# ==================================================

baseline_file = "Results/model_comparison.csv"

baseline = pd.read_csv(baseline_file)

print("----- Baseline Models -----")
print(baseline.to_string(index=False))


# ==================================================
# TUNED MODEL RESULTS
# ==================================================

tuned = pd.DataFrame({
    "Model": [
        "Tuned Logistic Regression",
        "Tuned Random Forest",
        "Tuned Gradient Boosting"
    ],

    "Accuracy": [
        0.8027,
        0.7885,
        0.7949
    ],

    "Precision": [
        0.6509,
        0.6696,
        0.6534
    ],

    "Recall": [
        0.5535,
        0.4011,
        0.4840
    ],

    "F1 Score": [
        0.5983,
        0.5017,
        0.5561
    ],

    "ROC-AUC": [
        0.8410,
        0.8408,
        0.8462
    ]
})


print("\n----- Tuned Models -----")
print(tuned.to_string(index=False))


# ==================================================
# COMBINE ALL RESULTS
# ==================================================

final_results = pd.concat(
    [
        baseline,
        tuned
    ],
    ignore_index=True
)


# ==================================================
# SORT BY ROC-AUC
# ==================================================

final_results = final_results.sort_values(
    by="ROC-AUC",
    ascending=False
)


# ==================================================
# ROUND VALUES
# ==================================================

metric_columns = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score",
    "ROC-AUC"
]

final_results[metric_columns] = (
    final_results[metric_columns]
    .round(4)
)


# ==================================================
# DISPLAY FINAL TABLE
# ==================================================

print("\n========== FINAL MODEL COMPARISON ==========\n")

print(
    final_results.to_string(index=False)
)


# ==================================================
# SAVE FINAL REPORT
# ==================================================

output_file = "Results/final_model_performance.csv"

final_results.to_csv(
    output_file,
    index=False
)

print(
    "\nFinal model performance report saved successfully."
)

print(
    "Output file:",
    output_file
)

print(
    "\n========== FINAL MODEL REPORT COMPLETED =========="
)