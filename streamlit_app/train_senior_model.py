import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib

# Create dataset manually
data = {
    "age": [20, 25, 30, 40, 50, 55, 60, 65, 70, 75, 80],
    "senior": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

X = df[["age"]]
y = df["senior"]

# Train model
model = DecisionTreeClassifier()
model.fit(X, y)

# Save model
joblib.dump(model, "senior_model.pkl")

print("✅ senior_model.pkl saved!")