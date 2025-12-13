import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

data = pd.read_csv(r"D:\Machine Learning\Datasets\Logistic-Regression\framingham.csv")
df = pd.DataFrame(data)

# THIS DATA SET INTITALLY CONTAINS MISSING VALUES
cleaned_data = df.copy()
for col in cleaned_data.columns:
    cleaned_data[col] = cleaned_data[col].fillna(cleaned_data[col].median())

# ------------------------ Model Generation ------------------------ #
x = cleaned_data.drop("TenYearCHD", axis=1)
y = cleaned_data["TenYearCHD"]

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.2, stratify=y)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

model = LogisticRegression(class_weight="balanced", max_iter=500)
model.fit(x_train_scaled, y_train)

threshold =0.3

y_pred = model.predict(x_test_scaled)
y_proba = model.predict_proba(x_test_scaled)[:,1]
y_pred_custom = (y_proba >= threshold).astype(int)

# Evaluation Metrics
acc = accuracy_score(y_test, y_pred_custom)
prec = precision_score(y_test, y_pred_custom, zero_division=0)
rec = recall_score(y_test, y_pred_custom)
f1 = f1_score(y_test, y_pred_custom)
cm = confusion_matrix(y_test, y_pred_custom)

print(f"Threshold = {threshold}")
print("Accuracy:", round(acc,4))
print("Precision:", round(prec,4))
print("Recall:", round(rec,4))
print("F1 Score:", round(f1,4))
print("Confusion Matrix:\n", cm)

# ---------- Precision-Recall curve ----------
precision_vals, recall_vals, pr_thresholds = precision_recall_curve(y_test, y_proba)
pr_auc = auc(recall_vals, precision_vals)

plt.figure(figsize=(7,5))
plt.plot(recall_vals, precision_vals, color='green', lw=2, label=f'PR AUC = {pr_auc:.3f}')
# mark chosen threshold point on the curve (pr_thresholds length = len(precision_vals)-1)
if len(pr_thresholds) > 0:
    idx = np.argmin(np.abs(pr_thresholds - threshold))
    plt.scatter(recall_vals[idx], precision_vals[idx], color='red', s=80, zorder=5)
    plt.annotate(f"thr={threshold}", (recall_vals[idx], precision_vals[idx]),
                 textcoords="offset points", xytext=(8,-8), fontsize=10)

plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('Precision-Recall Curve (Heart Disease)', fontsize=14)
plt.grid(alpha=0.3)
plt.legend(loc='best')
plt.tight_layout()
plt.show()