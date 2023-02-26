# U ovom primjeru poboljšavamo linearnu regresiju tako da funkciju aproksimiramo polinomom većeg stupnja
# Na taj se način može dobiti bolja greška, ali to također može dovesti i do dužeg vremena izvršavanja te overfittanja


import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import time

# Define the function to be approximated
def function(x):
    return np.sin(x + np.pi) - 1/(x**2 + 1) + x

# Generate some noisy training data
x_train = np.random.uniform(low=-10, high=10, size=(100,))
x_train.sort()
y_train = function(x_train) + np.random.normal(scale=0.1, size=(100,))

# Generate the test data
x_test = np.linspace(-10, 10, 20)
y_test = function(x_test)

# Fit a quadratic function to the data using polynomial regression
start_time = time.time()

degree = 3
poly_features = PolynomialFeatures(degree=degree)
X_train_poly = poly_features.fit_transform(x_train.reshape(-1, 1))
X_test_poly = poly_features.fit_transform(x_test.reshape(-1, 1))
model = LinearRegression()
model.fit(X_train_poly, y_train)

elapsed_time = time.time() - start_time

# Predict the test data
y_pred = model.predict(X_test_poly)

# Calculate the relative error
error = np.mean((y_test - y_pred)**2)

# Open file in "append" mode
with open('../vremena.txt', 'a') as file:
    # Write value to end of file
    file.write('Polynomial regression\n')
    file.write(f'Error: {error}\n')
    file.write(f'Elapsed time: {elapsed_time}\n')

# Plot the original function and the approximated function
plt.plot(x_test, y_pred, color='orange', label='Approximated function')
plt.plot(x_test, y_test, label='Original function')
plt.scatter(x_train, y_train, label='Training data')
plt.legend()
plt.show()
