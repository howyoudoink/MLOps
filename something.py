import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

# ----------------------------
# 1. LOAD DATA
# ----------------------------
df = pd.read_csv("D:\!MEHRAN\MLOps\Predictive Maintenance Dataset\\ai4i2020.csv")

# Drop irrelevant columns
df.drop(["UDI", "Product ID"], axis=1, inplace=True)

# Encode Machine Type
df["Type"] = df["Type"].map({"L": 0, "M": 1, "H": 2})

# Define features & target
X = df.drop("Machine failure", axis=1)
y = df["Machine failure"]

print("Class Distribution:")
print(y.value_counts())

# ----------------------------
# 2. TRAIN-TEST SPLIT
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ----------------------------
# 3. HANDLE CLASS IMBALANCE (SMOTE)
# ----------------------------
smote = SMOTE(random_state=42)

# ----------------------------
# 4. BUILD PIPELINE
# ----------------------------
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("smote", smote),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42
    ))
])

# ----------------------------
# 5. TRAIN MODEL
# ----------------------------
pipeline.fit(X_train, y_train)

# ----------------------------
# 6. EVALUATION
# ----------------------------
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

roc_auc = roc_auc_score(y_test, y_prob)
print("\nROC-AUC Score:", roc_auc)

# ----------------------------
# 7. CROSS VALIDATION
# ----------------------------
cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring="roc_auc")
print("\nCross-Validation ROC-AUC:", cv_scores.mean())

# ----------------------------
# 8. FEATURE IMPORTANCE
# ----------------------------
feature_importance = pipeline.named_steps["classifier"].feature_importances_
feature_names = X.columns

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": feature_importance
}).sort_values(by="Importance", ascending=False)

print("\nFeature Importance:")
print(importance_df)

# Plot feature importance
plt.figure(figsize=(10,6))
sns.barplot(data=importance_df, x="Importance", y="Feature")
plt.title("Feature Importance")
plt.tight_layout()
plt.savefig("feature_importance1.png")
plt.close()

# ----------------------------
# 9. SAVE MODEL
# ----------------------------
joblib.dump(pipeline, "automotive_maintenance_model1.pkl")

print("\nModel saved successfully!")