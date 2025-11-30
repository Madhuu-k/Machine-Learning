import numpy as np

x = np.array([1000, 1500, 2000, 2500, 3000]) # square foot
y = np.array([150000, 200000, 250000, 300000, 350000]) # price

x_mean = np.mean(x)
y_mean = np.mean(y)

m = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean)**2)  # m = covariance(x, y) / variance(x)
b = y_mean - m * x_mean # b = mean(y) - m * mean(x)


new_x = 2000
y_pred = m * new_x + b

MSE = (1/2) * np.mean((y - y_pred)**2)

print("Slope (m):", m)
print("Intercept (b):", b)
print("Predicted price for", new_x, "sq ft:", y_pred)
print("Mean Squared Error (MSE):", MSE)































