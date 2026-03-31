import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib

# Load dataset
data = pd.read_csv("sign_data.csv", header=None)

# Split features and labels
X = data.iloc[:, 1:]   # landmarks
y = data.iloc[:, 0]    # label (Hello, Yes, etc.)

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Accuracy
accuracy = model.score(X_test, y_test)
print(f"✅ Model Accuracy: {accuracy}")

# Save model
joblib.dump(model, "sign_model.pkl")
print("✅ sign_model.pkl saved!")