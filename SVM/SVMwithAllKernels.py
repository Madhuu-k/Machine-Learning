import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

data = {
    "Height_cm":      [190,185,170,200,165,178,210,195,182,172,205,168,188,199,180],
    "Weight_kg":      [110, 95, 70,130, 65, 80,140,120, 90, 68,135, 72,100,125, 85],
    "Bench_Max_kg":   [160,140,100,200, 90,120,220,180,135,105,210, 95,150,190,130],
    "Squat_Max_kg":   [200,180,120,240,110,150,260,220,170,130,250,120,190,230,160],
    "Training_Years": [10,  8,  3,  12,  2,  5,  14,  11,  7,  3,  13,  2,  9,  12,  6],
    "Strong":         [1,1,0,1,0,0,1,1,1,0,1,0,1,1,1]
}
df = pd.DataFrame(data)

x = df[['Height_cm','Weight_kg','Bench_Max_kg','Squat_Max_kg','Training_Years']]
y = df['Strong']

kernels = ['linear', 'poly', 'rbf']
c_values = [0.1, 1, 10]
gamma_values = ['scale', 0,1, 1]

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.2)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

for k in kernels:
    model = SVC(kernel=k, C=1.0, gamma='scale')
    model.fit(x_train_scaled, y_train)
    pred = model.predict(x_test_scaled)
    print(f"{k} Kernel -> Accuracy: ", accuracy_score(y_test, pred))