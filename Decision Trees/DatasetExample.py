import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split


data = pd.read_csv(r"D:\Machine Learning\Datasets\Decision-Tree\dataset_traffic_accident_prediction1.csv")
df = pd.DataFrame(data)

cleaned_data = df.copy()
categorical_cols = cleaned_data.select_dtypes(include="object").columns
numerical_cols = cleaned_data.select_dtypes(exclude="object").columns

numerical_cols = numerical_cols.drop("Accident")

cleaned_data[categorical_cols] = cleaned_data[categorical_cols].fillna("Unknown")

for col in numerical_cols:
    cleaned_data[col] = cleaned_data[col].fillna(cleaned_data[col].median())
    
cleaned_data = cleaned_data.dropna(subset=["Accident_Severity"])
x = cleaned_data.drop(["Accident", "Accident_Severity"], axis=1)
y = cleaned_data["Accident_Severity"]

# One Hot Encode Data -->
x_encoded = pd.get_dummies(x, drop_first=True)

x_train, x_test, y_train, y_test = train_test_split(x_encoded, y, random_state=4, test_size=0.2)

dt = DecisionTreeClassifier(
    max_depth=5,
    min_samples_leaf=3,
    random_state=42,
    class_weight="balanced"
)

dt.fit(x_train, y_train)

y_pred = dt.predict(x_test)

print("Predictions: ", y_pred[:10])

print("Accuracy: ", accuracy_score(y_test, y_pred))
print("Confusion Matrix: ", confusion_matrix(y_test, y_pred))
print("Classification Report: ", classification_report(y_test, y_pred))

plt.figure(figsize=(24, 24))
plot_tree(
    dt,
    feature_names=x_train.columns,
    class_names=dt.classes_,
    filled=True,
    rounded=True,
    fontsize=6
)
plt.show()
