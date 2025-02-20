# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Create the dataset
data = {
    "Experience": [0, 1, 2, 3, 4, 5, 6, 7],
    "Test Score": [85, 88, 92, 70, 75, 80, 95, 90],
    "Hired": [0, 0, 1, 0, 1, 1, 1, 1]
}
df = pd.DataFrame(data)

# Step 2: Define features and target variable
X = df[["Experience", "Test Score"]]
y = df["Hired"]

# Step 3: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 4: Train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)
 
# Step 5: Make predictions
y_pred = model.predict(X_test)

# Step 6: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 7: Predict for a new candidate
new_candidate = [[3, 85]]  # Example: 3 years of experience, test score of 85
hiring_decision = model.predict(new_candidate)
print("\nHiring Decision for new candidate (1=Hired, 0=Not Hired):", hiring_decision[0])