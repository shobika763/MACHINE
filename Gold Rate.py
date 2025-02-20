import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


data = pd.read_csv("C:/keerthana/Book2.csv")


le = LabelEncoder()
data['DateEncoded'] = le.fit_transform(data['Date'])


X = data[['DateEncoded']]
y = data['GoldRate']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)


predicted = model.predict(X_test)


print("Model Coefficient:", model.coef_[0])
print("Model Intercept:", model.intercept_)
print("Predictions for Test Data:")
for actual, pred in zip(y_test, predicted):
    print(f"Actual: {actual}, Predicted: {pred}")


plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, model.predict(X), color='red', label='Linear Fit')
plt.xlabel('Date Encoded')
plt.ylabel('Gold Rate')
plt.title('Gold Rate Prediction')
plt.legend()
plt.show()