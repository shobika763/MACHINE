# Import the library
from sklearn.linear_model import LinearRegression

# Step 1: Provide training data
X = [[1], [2], [3], [4], [5]]  # Input (e.g., numbers)
y = [2, 4, 6, 8, 10]           # Output (e.g., double the numbers)

# Step 2: Create and train the model
model = LinearRegression()  # Create a model
model.fit(X, y)             # Train the model with the data

# Step 3: Make a prediction
new_data = [[4]]                # A new number to predict (6)
prediction = model.predict(new_data)  # Model predicts its output
print(f"Predicted output for input {new_data[0][0]}: {prediction[0]:.2f}")