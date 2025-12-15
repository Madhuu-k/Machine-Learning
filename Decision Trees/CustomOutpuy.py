import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = {
    "age": [45,54,39,61,48,57,42,66,51,36,59,44,63,47,52,34,58,41,69,46],
    "cholesterol": [230,260,180,290,210,275,195,305,225,170,285,200,310,215,245,165,270,190,320,205],
    "bp": [130,140,120,150,128,145,118,155,135,110,148,125,160,130,138,108,142,120,165,128],
    "max_hr": [150,135,170,120,160,130,165,110,140,175,125,158,105,155,145,180,132,168,100,152],
    "smoker": [1,1,0,1,0,1,0,1,0,0,1,0,1,0,1,0,1,0,1,0],
    "diabetic": [0,1,0,1,0,0,0,1,1,0,1,0,1,0,0,0,1,0,1,0],
    "target": [1,1,0,1,0,1,0,1,1,0,1,0,1,0,1,0,1,0,1,0]
}

df = pd.DataFrame(data)

x = df[["age",  "cholesterol", "bp", "max_hr", "smoker", "diabetic"]].values
y = df["target"].map({
    0 : "False",
    1 : "True"
})

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=24, test_size=0.2, stratify=y)
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

dt = DecisionTreeClassifier(
    max_depth=4,
    min_samples_leaf=2,
    random_state=42
)

dt.fit(x_train_scaled, y_train)

y_pred = dt.predict(x_test_scaled)
print(y_pred)

print("Accuracy: ", accuracy_score(y_test, y_pred))
print("Confusion Matrix: ", confusion_matrix(y_test, y_pred))
print("Classification Report: ", classification_report(y_test, y_pred))