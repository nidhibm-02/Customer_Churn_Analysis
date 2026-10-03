import pandas as pd
import matplotlib.pyplot as plt

print("========== FINAL MODEL VISUALIZATION ==========\n")

# Load final model results
file_path = "Results/final_model_performance.csv"
df = pd.read_csv(file_path)

# --------------------------------------------------
# Plot 1: Accuracy
# --------------------------------------------------

plt.figure(figsize=(10, 6))

plt.bar(
    df["Model"],
    df["Accuracy"]
)

plt.title("Model Accuracy Comparison")
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.xticks(rotation=45, ha="right")
plt.ylim(0, 1)
plt.tight_layout()

plt.savefig(
    "Visualizations/final_model_accuracy.png",
    dpi=300
)

plt.close()


# --------------------------------------------------
# Plot 2: Precision, Recall and F1
# --------------------------------------------------

plt.figure(figsize=(12, 6))

x = range(len(df))

plt.plot(
    x,
    df["Precision"],
    marker="o",
    label="Precision"
)

plt.plot(
    x,
    df["Recall"],
    marker="o",
    label="Recall"
)

plt.plot(
    x,
    df["F1 Score"],
    marker="o",
    label="F1 Score"
)

plt.xticks(
    x,
    df["Model"],
    rotation=45,
    ha="right"
)

plt.xlabel("Model")
plt.ylabel("Score")
plt.title("Precision, Recall and F1 Score Comparison")
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()

plt.savefig(
    "Visualizations/final_model_metrics.png",
    dpi=300
)

plt.close()


# --------------------------------------------------
# Plot 3: ROC-AUC
# --------------------------------------------------

plt.figure(figsize=(10, 6))

plt.bar(
    df["Model"],
    df["ROC-AUC"]
)

plt.title("ROC-AUC Comparison")
plt.xlabel("Model")
plt.ylabel("ROC-AUC")
plt.xticks(rotation=45, ha="right")
plt.ylim(0, 1)
plt.tight_layout()

plt.savefig(
    "Visualizations/final_model_roc_auc.png",
    dpi=300
)

plt.close()


print("Final model visualizations saved successfully.")

print("\nSaved files:")
print("Visualizations/final_model_accuracy.png")
print("Visualizations/final_model_metrics.png")
print("Visualizations/final_model_roc_auc.png")

print("\n========== FINAL MODEL VISUALIZATION COMPLETED ==========")