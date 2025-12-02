import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv(r"D:\Machine Learning\Datasets\Linear-Regression\house_price_regression_dataset.csv")

x = data[['Square_Footage', 'Num_Bedrooms', 'Num_Bathrooms', 'Lot_Size', 'Garage_Size', 'Neighborhood_Quality']]
y = data['House_Price']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test) 

# Evaluate the model
RMSE = mean_squared_error(y_test, y_pred)
R2 = r2_score(y_test, y_pred)

print("RMSE:", RMSE)
print("R²:", R2)
print("\nCoefficients:", model.coef_)
print("Intercept:", model.intercept_)