import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib

# Load dataset
data = pd.read_csv("hair_data.csv")

X = data[["age", "hair_length"]]
y = data["gender"]

# Train model
model = DecisionTreeClassifier()
model.fit(X, y)

# Save model
joblib.dump(model, "hair_model.pkl")

print("✅ hair_model.pkl saved!")