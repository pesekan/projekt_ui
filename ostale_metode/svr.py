import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt

# Define the function to approximate
def f(x):
    return np.sin(x + np.pi) - 1/(x**2 + 1) + x

# Generate training data
X_train = np.random.uniform(-10, 10, size=100).reshape(-1, 1)
y_train = f(X_train)

# Define the SVM model
model = SVR(kernel='rbf', C=100, gamma=0.1)

# Train the model
model.fit(X_train, y_train)

# Generate test data
X_test = np.linspace(-10, 10, num=20).reshape(-1, 1)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Plot the results
plt.plot(X_test, y_pred, label='SVM')
plt.plot(X_test, f(X_test), label='true')
plt.legend()
plt.show()
