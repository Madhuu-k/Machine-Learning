import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


data = pd.read_csv(r"D:\Machine Learning\Datasets\Decision-Tree\dataset_traffic_accident_prediction1.csv")
df = pd.DataFrame(data)

cleaned_data = df.copy()
categorical_cols = cleaned_data.select_dtypes(include="object").columns
numerical_cols = cleaned_data.select_dtypes(exclude="object").columns

numerical_cols = numerical_cols.drop("Accident")

cleaned_data[categorical_cols] = cleaned_data[categorical_cols].fillna("Unknown")
cleaned_data = cleaned_data[cleaned_data["Accident_Severity"] != "Unknown"]


for col in numerical_cols:
    cleaned_data[col] = cleaned_data[col].fillna(cleaned_data[col].median())
    
cleaned_data = cleaned_data.dropna(subset=["Accident_Severity"])
x = cleaned_data.drop(["Accident", "Accident_Severity"], axis=1)
y = cleaned_data["Accident_Severity"]

# One Hot Encode Data -->
x_encoded = pd.get_dummies(x, drop_first=True)

x_train, x_test, y_train, y_test = train_test_split(x_encoded, y, random_state=4, test_size=0.2)

rt = RandomForestClassifier(
    max_depth=6,
    min_samples_leaf=3,
    n_estimators=200,
    n_jobs=1,
    random_state=42,
    class_weight="balanced"
)

rt.fit(x_train, y_train)

y_pred = rt.predict(x_test)

print("Predictions: ", y_pred[:10])

print("Accuracy: ", accuracy_score(y_test, y_pred))
print("Confusion Matrix: ", confusion_matrix(y_test, y_pred))
print("Classification Report: ", classification_report(y_test, y_pred))

importance = pd.Series(
    rt.feature_importances_,
    index = x_train.columns
).sort_values(ascending=False)

print("Top Contributors: ")
print(importance.head(10))

