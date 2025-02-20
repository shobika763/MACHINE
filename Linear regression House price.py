# Importing the necessary libraries
import numpy as np
from sklearn.linear_model import LinearRegression

# Data: House Size (in square feet), Number of Bedrooms, and Price (in dollars)
# Features: [Size (sq ft), Bedrooms]
X = np.array([[1400, 3], [1600, 3], [1700, 3], [1800, 3], [1100, 2],
              [1900, 4], [2000, 4], [2100, 4], [1500, 3], [1600, 3]])

# Target variable: Price
y = np.array([400000, 450000, 475000, 500000, 300000,
              550000, 600000, 620000, 450000, 470000])

# Initializing the multiple linear regression model
model = LinearRegression()

# Training the model on the data
model.fit(X, y)

# Output the coefficients (weights for each feature) and the intercept
print("Coefficients (for Size and Bedrooms):", model.coef_)


# Prediction: Price of a new house with 1500 sq ft and 3 bedrooms
new_house = np.array([[1500, 3]])
predicted_price = model.predict(new_house)
print(f"Predicted price for a 1500 sq ft, 3 bedroom house: ${predicted_price[0]:,.2f}")
