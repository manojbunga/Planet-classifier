import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv('data/planets.csv')

# Basic preprocessing
X = data[['mass', 'distance', 'radius']]
y = data['type']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.2f}")

# Plot feature importance
features = X.columns
importance = model.feature_importances_
plt.bar(features, importance)
plt.title("Feature Importance")
plt.show()