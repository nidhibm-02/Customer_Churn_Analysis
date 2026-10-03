import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

print("========== MODEL HYPERPARAMETER TUNING ==========\n")

# Load cleaned dataset
file_path = "Dataset/telco_churn_cleaned.csv"
df = pd.read_csv(file_path)

# Remove customer ID
df = df.drop("customerID", axis=1)

# Separate features and target
X = df.drop("Churn", axis=1)
y = df["Churn"].map({
    "No": 0,
    "Yes": 1
})

# One-hot encoding
X = pd.get_dummies(X, drop_first=True, dtype=int)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Scaling for Logistic Regression
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==================================================
# 1. LOGISTIC REGRESSION TUNING
# ==================================================

print("----- Logistic Regression Tuning -----")

lr = LogisticRegression(
    max_iter=1000,
    random_state=42
)

lr_params = {
    "C": [0.01, 0.1, 1, 10],
    "solver": ["liblinear", "lbfgs"]
}

lr_grid = GridSearchCV(
    lr,
    lr_params,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1
)

lr_grid.fit(X_train_scaled, y_train)

best_lr = lr_grid.best_estimator_

print("Best Parameters:", lr_grid.best_params_)
print("Best CV ROC-AUC:", round(lr_grid.best_score_, 4))

# Evaluation
y_pred = best_lr.predict(X_test_scaled)
y_prob = best_lr.predict_proba(X_test_scaled)[:, 1]

print("\nTest Set Results:")
print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score : {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC  : {roc_auc_score(y_test, y_prob):.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# ==================================================
# 2. RANDOM FOREST TUNING
# ==================================================

print("\n----- Random Forest Tuning -----")

rf = RandomForestClassifier(
    random_state=42,
    n_jobs=-1
)

rf_params = {
    "n_estimators": [100, 200],
    "max_depth": [5, 10, 15],
    "min_samples_split": [2, 5]
}

rf_grid = GridSearchCV(
    rf,
    rf_params,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1
)

rf_grid.fit(X_train, y_train)

best_rf = rf_grid.best_estimator_

print("Best Parameters:", rf_grid.best_params_)
print("Best CV ROC-AUC:", round(rf_grid.best_score_, 4))

# Evaluation
y_pred = best_rf.predict(X_test)
y_prob = best_rf.predict_proba(X_test)[:, 1]

print("\nTest Set Results:")
print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score : {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC  : {roc_auc_score(y_test, y_prob):.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# ==================================================
# 3. GRADIENT BOOSTING TUNING
# ==================================================

print("\n----- Gradient Boosting Tuning -----")

gb = GradientBoostingClassifier(
    random_state=42
)

gb_params = {
    "n_estimators": [50, 100, 150],
    "learning_rate": [0.05, 0.1],
    "max_depth": [2, 3]
}

gb_grid = GridSearchCV(
    gb,
    gb_params,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1
)

gb_grid.fit(X_train, y_train)

best_gb = gb_grid.best_estimator_

print("Best Parameters:", gb_grid.best_params_)
print("Best CV ROC-AUC:", round(gb_grid.best_score_, 4))

# Evaluation
y_pred = best_gb.predict(X_test)
y_prob = best_gb.predict_proba(X_test)[:, 1]

print("\nTest Set Results:")
print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score : {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC  : {roc_auc_score(y_test, y_prob):.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\n========== HYPERPARAMETER TUNING COMPLETED ==========")