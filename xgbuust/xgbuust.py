import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import time

# Define the function to approximate
def f(x):
    return np.sin(x + np.pi) - 1/(x**2 + 1) + x

# Generate training data
X_train = np.random.uniform(-10, 10, size=100).reshape(-1, 1)
y_train = f(X_train) + np.random.normal(scale=0.1, size=(100, 1))

# Define the xgbuust model
start_time = time.time()

model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=5)

# Train the model
model.fit(X_train, y_train)

elapsed_time = time.time() - start_time

# Generate test data
X_test = np.linspace(-10, 10, 20).reshape(-1, 1)

# Make predictions on the test data
y_test = f(X_test)
y_pred = model.predict(X_test)

error = 0
for i in range(len(y_test)):
    error += (y_test[i] - y_pred[i])**2
error /= len(y_test)

str_error = str(error)
str_error = str_error.replace("[","")
str_error = str_error.replace("]","")

# Open file in "append" mode
with open('../vremena.txt', 'a') as file:
    # Write value to end of file
    file.write('XGBoost\n')
    file.write(f'Error: {str_error}\n')
    file.write(f'Elapsed time: {elapsed_time}\n')

# Plot the results
plt.scatter(X_train, y_train, label='Training data')
plt.plot(X_test, y_pred, color='orange', label='xgboost')
plt.plot(X_test, f(X_test), label='true')
plt.legend()
plt.show()
