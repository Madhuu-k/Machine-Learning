import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

data = {
    'Horsepower': [120, 150, 200, 250, 300, 350, 400, 450, 500, 550],
    'Engine_Size': [1.6, 2.0, 2.5, 3.0, 3.5, 4.0, 4.2, 4.5, 5.0, 5.5],
    'Weight': [1100, 1300, 1500, 1600, 1800, 2000, 2200, 2400, 2600, 2800],
    'Age': [10, 8, 6, 5, 4, 3, 2, 2, 1, 1],
    'Car_Price': [5000, 7000, 9500, 13000, 16000, 20000, 24000, 28000, 33000, 38000]
}

df = pd.DataFrame(data)
print(df)
x = df[['Horsepower', 'Engine_Size', 'Weight', 'Age']]
y = df['Car_Price']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

RMSE = mean_squared_error(y_test, y_pred)
R2 = r2_score(y_test, y_pred)

print("RMSE:", RMSE)
print("R²:", R2)
print("\nCoefficients:", model.coef_)
print("Intercept:", model.intercept_)