import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# Define the function to be approximated
def func(x):
    return np.sin(x + np.pi) - 1/(x**2 + 1) + x

# Generate training data
np.random.seed(0)
X_train = np.sort(np.random.uniform(low=-10, high=10, size=(200, 1)), axis=0)
y_train = func(X_train).ravel()

# Generate test data
X_test = np.arange(-10.0, 10.0, 0.1)[:, np.newaxis]

# Train the Random Forest regression model
rf = RandomForestRegressor(n_estimators=100)
rf.fit(X_train, y_train)

# Predict the output values for the test data
y_pred = rf.predict(X_test)

# Plot the results
plt.figure(figsize=(10, 5))
plt.scatter(X_train, y_train, color='black', label='training data')
plt.plot(X_test, y_pred, color='blue', label='Random Forest')
plt.plot(X_test, func(X_test), color='red', linestyle='--', label='true function')
plt.legend(loc='lower left')
plt.title('Random Forest Regression')
plt.show()
