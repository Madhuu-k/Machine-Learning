import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix
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

x_test_scaled = scaler.fit_transform(x_test)
x_train_sclaed = scaler.fit_transform(x_train)

knn = KNeighborsRegressor(n_neighbors=3)
knn.fit(x_train_sclaed, y_train)

y_pred = knn.predict(x_test_scaled)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("Predictions:", y_pred)
print("Actual:", y_test.values)
print("RMSE:", rmse)