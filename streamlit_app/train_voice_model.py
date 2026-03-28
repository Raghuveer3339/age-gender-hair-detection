import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib

# Load dataset
data = pd.read_csv("voice_data.csv")

X = data.drop("label", axis=1)
y = data["label"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Accuracy
accuracy = model.score(X_test, y_test)
print("✅ Model Accuracy:", accuracy)

# Save model
joblib.dump(model, "voice_model.pkl")

print("✅ voice_model.pkl saved!")