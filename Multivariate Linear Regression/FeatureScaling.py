import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = {
    'WWE_Title_Reigns':            [14, 8, 6, 10, 7, 2, 0, 0, 0, 1],
    'World_Heavyweight_Reigns':    [5, 4, 0, 0, 1, 0, 0, 0, 0, 0],
    'Universal_Title_Reigns':      [2, 0, 0, 0, 2, 0, 0, 2, 0, 1],
    'Intercontinental_Reigns':     [0, 0, 2, 0, 0, 1, 3, 0, 1, 2],
    'US_Title_Reigns':             [0, 0, 3, 5, 2, 3, 2, 0, 0, 1],
    'Raw_Tag_Reigns':              [3, 6, 0, 1, 0, 1, 5, 0, 4, 2],
    'Smackdown_Tag_Reigns':        [0, 0, 0, 0, 0, 4, 2, 1, 0, 0],
    'Years_in_WWE':                [20, 18, 15, 16, 12, 10, 8, 6, 5, 7],
    'Big_Matches_Won':             [95, 88, 75, 80, 74, 60, 58, 52, 40, 45],
    'Popularity_Score':            [100, 97, 92, 94, 90, 85, 82, 80, 75, 78]
}

df = pd.DataFrame(data)

x = df[['WWE_Title_Reigns', 'World_Heavyweight_Reigns', 'Universal_Title_Reigns',
        'Intercontinental_Reigns', 'US_Title_Reigns', 'Raw_Tag_Reigns',
        'Smackdown_Tag_Reigns', 'Years_in_WWE', 'Big_Matches_Won']]

y = df['Popularity_Score']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
scaler.fit(x_train)

x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)

model = LinearRegression()
model.fit(x_train_scaled, y_train)

y_pred = model.predict(x_test_scaled)

rmse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("RMSE:", rmse)
print("R²:", r2)

features = x.columns.tolist()
coefs = model.coef_
for f, c in zip(features, coefs):
    print(f"{f}: {c:.3f}")
print("Intercept:", model.intercept_)
