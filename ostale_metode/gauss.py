import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import matplotlib.pyplot as plt

# Define the function to be approximated
def func(x):
    return np.sin(x + np.pi) - 1/(x**2 + 1) + x

# Generate training data
np.random.seed(0)
X_train = np.sort(np.random.uniform(low=-10, high=10, size=(100, 1)), axis=0)
y_train = func(X_train).ravel()

# Generate test data
X_test = np.sort(np.random.uniform(low=-10, high=10, size=(20, 1)), axis=0)

# Define the kernel function
kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))

# Train the Gaussian process regression model
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
gp.fit(X_train, y_train)

# Predict the output values for the test data
y_pred, std_pred = gp.predict(X_test, return_std=True)

# Plot the results
plt.figure(figsize=(10, 5))
plt.scatter(X_train, y_train, color='black', label='training data')
plt.plot(X_test, y_pred, color='blue', label='GPR')
plt.fill_between(X_test[:, 0], y_pred - 1.96 * std_pred, y_pred + 1.96 * std_pred,
                 alpha=0.2, color='blue')
plt.plot(X_train, func(X_train), color='red', linestyle='--', label='true function')
plt.legend(loc='lower left')
plt.title('Gaussian Process Regression')
plt.show()
