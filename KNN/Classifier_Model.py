import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

data = {
    'Study_Hours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Attendance':  [50, 55, 60, 65, 70, 75, 80, 82, 90, 95],
    'Pass':        [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

x = df[['Study_Hours', 'Attendance']]
y = df['Pass']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train_scaled, y_train)

y_pred = knn.predict(x_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1  = f1_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
confusionMatrix = confusion_matrix(y_test, y_pred)

knn_manhattan = KNeighborsClassifier(n_neighbors=3, p=1)
knn_manhattan.fit(x_train_scaled, y_train)

y_pred_manhattan = knn_manhattan.predict(x_test_scaled)

print("\nUsing Manhattan Distance (p=1):")
print("Accuracy:", accuracy_score(y_test, y_pred_manhattan))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_manhattan))

k_values = np.arange(1, 11)
k_scores = []

for k in k_values:
     knn_temp = KNeighborsClassifier(n_neighbors=k)
     scores = cross_val_score(knn_temp, x_train_scaled, y_train, cv=4, scoring='accuracy')
     k_scores.append(scores.mean())
     
print("K values tested: ", list(k_values))
print("cross value score: ", np.round(k_scores, 3))

best_k = k_values[np.argmax(k_scores)]
print("Best K value: ", best_k)
