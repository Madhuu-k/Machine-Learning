import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score

data = {
    "Height_cm":      [190, 185, 170, 200, 165, 178, 210, 195, 182, 172, 205, 168, 188, 199, 180],
    "Weight_kg":      [110,  95,  70, 130,  65,  80, 140, 120,  90,  68, 135,  72, 100, 125,  85],
    "Bench_Max_kg":   [160, 140, 100, 200,  90, 120, 220, 180, 135, 105, 210,  95, 150, 190, 130],
    "Squat_Max_kg":   [200, 180, 120, 240, 110, 150, 260, 220, 170, 130, 250, 120, 190, 230, 160],
    "Training_Years": [10,   8,   3,   12,   2,   5,   14,   11,   7,   3,   13,   2,   9,   12,   6],

    # 1 = Strong Wrestler, 0 = Not Strong
    "Strong":         [1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1]
}

df = pd.DataFrame(data)

x = df[["Height_cm", "Weight_kg", "Bench_Max_kg", "Squat_Max_kg", "Training_Years"]]
y = df["Strong"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.fit_transform(x_test)

# ----------------- KNN Classifier ----------------- #

model = KNeighborsClassifier(n_neighbors=3)
model.fit(x_train_scaled, y_train)

y_pred = model.predict(x_test_scaled)

accracy = accuracy_score(y_test, y_pred)
confusionMatrix = confusion_matrix(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print("Predictions:", y_pred)
print("Actual:", y_test.values)
print("Accuracy:", accracy)
print("Confusion Matrix:\n", confusionMatrix)

# ----------------- KNN Regressor ----------------- #

model_reg = KNeighborsRegressor(n_neighbors=3)
model_reg.fit(x_train_scaled, y_train)

y_pred_reg = model_reg.predict(x_test_scaled)

mse_reg = mean_squared_error(y_test, y_pred_reg)
rmse_reg = np.sqrt(mse_reg)
r2 = r2_score(y_test, y_pred_reg)
accruracy = accuracy_score(y_test, np.round(y_pred_reg))
confusionMatrix = confusion_matrix(y_test, np.round(y_pred_reg))

print("\n\nAccuracy (Regressor):", accruracy)
print("Confusion Matrix (Regressor):\n", confusionMatrix)
print("RMSE (Regressor):", rmse_reg)
print("R2 Score (Regressor):", r2)
print("Predictions (Regressor):", y_pred_reg)