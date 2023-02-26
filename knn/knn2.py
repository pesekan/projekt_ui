# U ovom primjeru poboljšavamo knn metodu skaliranjem input featurea.
# Prvo prilagođavamo scaler na podatke o obuci i koristimo ga za transformaciju training i test podataka, a zatim
# obučavamo KNN model na skaliranim podacima obuke i procjenjujemo njegovu izvedbu na skaliranim testnim podacima.
# To može pomoći u poboljšanju izvedbe KNN modela smanjenjem utjecaja značajki s različitim mjerilima.


import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
import time

# Define the function to be approximated
def func(x):
    return np.sin(x + np.pi) - 1 / (x**2 + 1) + x

# Generate the training data
X_train = np.random.uniform(low=-10, high=10, size=(100, 1))
y_train = func(X_train) + np.random.normal(scale=0.1, size=(100, 1))

# Generate the test data
X_test = np.linspace(-10, 10, 20).reshape(-1, 1)
y_test = func(X_test)

# Scale the input features
start_time = time.time()

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the KNN model
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

elapsed_time = time.time() - start_time

# Evaluate the model
y_pred = knn.predict(X_test_scaled)
mse = np.mean((y_pred - y_test)**2)

# Open file in "append" mode
with open('../vremena.txt', 'a') as file:
    # Write value to end of file
    file.write('KNN with scaled input features\n')
    file.write(f'Error: {mse}\n')
    file.write(f'Elapsed time: {elapsed_time}\n')

import matplotlib.pyplot as plt
# Plot the results
plt.scatter(X_train, y_train, label='training data')
plt.plot(X_test, y_pred, color='orange', label='KNN regression')
plt.plot(X_test, func(X_test), label='true function')
plt.legend()
plt.title(f'KNN Regression (n_neighbors=5)')
plt.show()
