import numpy as np
import pandas as pd

data = pd.read_csv(r"D:\Machine Learning\Datasets\Linear-Regression\house_price_regression_dataset.csv")

x = data['Square_Footage'].values
y = data['House_Price'].values

mean_x, mean_y = np.mean(x), np.mean(y)

m = np.sum((x - mean_x) * (y - mean_y)) / np.sum((x - mean_x)**2)

b = mean_y - m * mean_x

y_pred = m * x + b

new_x = 2000
newY_pred = m * new_x + b

MSE = 0.5 * np.mean((y - y_pred)**2)

print("Slope m:", m)
print("Intercept b:", b)
print("Predicted price for 2000 sqft:", newY_pred)
print("MSE:", MSE)
