import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

data = {
    'Study_Hours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Attendance':  [50, 55, 60, 65, 70, 75, 80, 82, 90, 95],
    'Pass':        [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]   # 1 = pass, 0 = fail
}

df = pd.DataFrame(data)

x = df[['Study_Hours', 'Attendance']]
y = df['Pass']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.fit_transform(x_test)

model = LogisticRegression()
model.fit(x_train_scaled, y_train)

y_pred = model.predict(x_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
confusionMatrix = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("F1 Score:", f1)
print("Recall:", recall)    
print("Confusion Matrix:\n", confusionMatrix)

probas = model.predict_proba(x_test_scaled)
print("Predicted Probabilities:\n", probas)

y_pred_custom =  (probas[:,1] >= 0.6).astype(int)
print("\nPredictions with threshold 0.6:", y_pred_custom)