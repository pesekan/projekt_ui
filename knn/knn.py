import numpy as np
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
import time

# Define the function to be approximated
def func(x):
    return np.sin(x + np.pi) - 1/(x**2 + 1) + x

# Generate training data
np.random.seed(0)
X_train = np.sort(np.random.uniform(low=-10, high=10, size=(100, 1)), axis=0)
y_train = func(X_train) + np.random.normal(scale=0.1, size=(100, 1))

# Generate test data
X_test = np.linspace(-10, 10, 20).reshape(-1, 1)

# Train the KNN regression model
start_time = time.time()

n_neighbors = 5
knn = KNeighborsRegressor(n_neighbors=n_neighbors)
knn.fit(X_train, y_train)

elapsed_time = time.time() - start_time

# Predict the output values for the test data
y_test = func(X_test)
y_pred = knn.predict(X_test)
error = np.mean((y_test - y_pred)**2)

# Open file in "append" mode
with open('../vremena.txt', 'a') as file:
    # Write value to end of file
    file.write('KNN\n')
    file.write(f'Error: {error}\n')
    file.write(f'Elapsed time: {elapsed_time}\n')

# Plot the results
plt.scatter(X_train, y_train, label='training data')
plt.plot(X_test, y_pred, color='orange', label='KNN regression')
plt.plot(X_test, func(X_test), label='true function')
plt.legend()
plt.title(f'KNN Regression (n_neighbors={n_neighbors})')
plt.show()
