from sklearn.linear_model import LinearRegression
import numpy as np

# Data: [House Size (in sq ft)], [Price (in $)]
X = np.array([[1500], [2000], [2500], [3000]])  # Features
y = np.array([300000, 400000, 500000, 600000])  # Target

# Create and train the model
model = LinearRegression()
model.fit(X, y)

# Predict price for a 2700 sq ft house
predicted_price = model.predict([[3000]])
print(f"Predicted Price: ${predicted_price[0]:.2f}")