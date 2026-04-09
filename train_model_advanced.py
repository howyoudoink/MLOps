import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.linear_model import LogisticRegression
from imblearn.pipeline import Pipeline

# ==========================================================
# 1. LOAD DATA
# ==========================================================

df = pd.read_csv(r"D:\!MEHRAN\MLOps\Predictive Maintenance Dataset\ai4i2020.csv")

df.columns = df.columns.astype(str)

# Drop leakage columns
df.drop(["UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"], axis=1, inplace=True)

# Encode categorical
df["Type"] = df["Type"].map({"L": 0, "M": 1, "H": 2})

# ==========================================================
# 2. FEATURE ENGINEERING (SCALED PROPERLY)
# ==========================================================

df["temp_diff"] = df["Process temperature [K]"] - df["Air temperature [K]"]

df["power"] = (df["Rotational speed [rpm]"] * df["Torque [Nm]"]) / 1000

df["wear_torque"] = (df["Tool wear [min]"] * df["Torque [Nm]"]) / 100

# ==========================================================
# 3. FEATURES & TARGET
# ==========================================================

X = df.drop("Machine failure", axis=1)
y = df["Machine failure"]

print("\nClass Distribution:")
print(y.value_counts())

# ==========================================================
# 4. SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==========================================================
# 5. MODEL (FINAL FIX)
# ==========================================================

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", model)
])

# ==========================================================
# 6. TRAIN
# ==========================================================

pipeline.fit(X_train, y_train)

# ==========================================================
# 7. EVALUATION
# ==========================================================

y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nROC-AUC:", roc_auc_score(y_test, y_prob))

# ==========================================================
# 8. CROSS VALIDATION
# ==========================================================

cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring="roc_auc")
print("\nCross Val ROC-AUC:", cv_scores.mean())

# ==========================================================
# 9. FEATURE IMPORTANCE (COEFFICIENTS)
# ==========================================================

coefficients = pipeline.named_steps["classifier"].coef_[0]

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": coefficients
}).sort_values(by="Importance", key=abs, ascending=False)

print("\nFeature Importance (Logistic Coefficients):")
print(importance_df)

plt.figure(figsize=(10,6))
sns.barplot(data=importance_df, x="Importance", y="Feature")
plt.title("Feature Importance")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.close()

# ==========================================================
# 10. SAVE MODEL
# ==========================================================

joblib.dump(pipeline, "automotive_model_final.pkl")

print("\n✅ Model saved as automotive_model_final.pkl")