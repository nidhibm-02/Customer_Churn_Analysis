import pandas as pd
import matplotlib.pyplot as plt

print("========== MODEL COMPARISON ==========\n")

# Model results
results = {
    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "Gradient Boosting",
        "SVM",
        "KNN",
        "Naive Bayes"
    ],
    "Accuracy": [
        0.8070,
        0.7942,
        0.8048,
        0.7984,
        0.7928,
        0.7473,
        0.6558
    ],
    "Precision": [
        0.6584,
        0.6312,
        0.6623,
        0.6552,
        0.6444,
        0.5253,
        0.4269
    ],
    "Recall": [
        0.5668,
        0.5401,
        0.5401,
        0.5080,
        0.4893,
        0.5000,
        0.8663
    ],
    "F1 Score": [
        0.6092,
        0.5821,
        0.5950,
        0.5723,
        0.5562,
        0.5123,
        0.5719
    ],
    "ROC-AUC": [
        0.8418,
        0.8267,
        0.8419,
        0.8421,
        0.7961,
        0.7718,
        0.8092
    ]
}

# Create DataFrame
results_df = pd.DataFrame(results)

# Display results
print(results_df.to_string(index=False))

# Save results
output_file = "Results/model_comparison.csv"
results_df.to_csv(output_file, index=False)

print(f"\nModel comparison saved to: {output_file}")

# Create comparison chart
metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score",
    "ROC-AUC"
]

results_df.set_index("Model")[metrics].plot(
    kind="bar",
    figsize=(12, 7)
)

plt.title("Machine Learning Model Performance Comparison")
plt.ylabel("Score")
plt.xlabel("Model")
plt.ylim(0, 1)
plt.xticks(rotation=30)
plt.legend(title="Metrics")
plt.tight_layout()

chart_file = "Visualizations/model_comparison.png"
plt.savefig(chart_file, dpi=300)
plt.close()

print(f"Comparison chart saved to: {chart_file}")

print("\nModel comparison completed!")