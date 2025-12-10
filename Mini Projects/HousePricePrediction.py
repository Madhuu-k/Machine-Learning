# HOUSE PRICE PREDICTION USING LINEAR REGRESSION
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data = pd.read_csv(r"D:\Machine Learning\Datasets\Linear-Regression\Housing.csv")
x = data[[
    "bedrooms", 
    "bathrooms", 
    "sqft_living", 
    "sqft_lot", 
    "floors", 
    "waterfront",
    "view",
    "condition",
    "grade",
    "sqft_above",
    "sqft_basement",
    "yr_built",
    "yr_renovated",
    "lat",
    "long", 
    "sqft_living15",
    "sqft_lot15"
]].values

y = data["price"].values

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.2)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.fit_transform(x_test)

model = LinearRegression()
model.fit(x_train_scaled, y_train)

y_pred = model.predict(x_test_scaled)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error: ", mse)
print("Root Mean Square Error: ", rmse)
print("R2 Score: ", r2)

plt.scatter(y_test, y_pred, alpha=0.7, color="red")
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("House Price Prediction With Linear Regression")
plt.grid(color="black", linestyle="--", linewidth=0.5)
plt.show()