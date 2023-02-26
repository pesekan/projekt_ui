# U ovom primjeru poboljšavamo neuronsku mrežu tako što dodajemo još slojeva neurona u model
# Može se primijetiti da svaki idući sloj neurona sadrži manje neurona od prethodnog sloja kako bismo smanjili
# dimenzionalnost problema


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time

# Define the function to be approximated
def function(x):
    return np.sin(x + np.pi) - 1/(x**2 + 1) + x

# Generate some noisy training data
x = np.random.uniform(low=-10, high=10, size=(120,))
x.sort()
y_noisy = function(x) + np.random.normal(scale=0.1, size=(120,))

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y_noisy, test_size=0.16667)

# Create the model
start_time = time.time()

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(64, input_dim=1, activation='relu'))
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(8, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='linear'))

# Compile the model
model.compile(loss='mse', optimizer='adam')

# Train the model
model.fit(x_train, y_train, epochs=1000, verbose=0)
elapsed_time = time.time() - start_time

# Evaluate the model
loss = model.evaluate(x_test, y_test)

# Open file in "append" mode
with open('../vremena.txt', 'a') as file:
    # Write value to end of file
    file.write('Neural network with multiple layers\n')
    file.write(f'Error: {loss}\n')
    file.write(f'Elapsed time: {elapsed_time}\n')

# Plot the original function and the approximated function
plt.scatter(x, y_noisy, label='Noisy data')
plt.plot(x, model.predict(x), color='orange', label='Approximated function')
plt.plot(x, function(x), label='Original function')
plt.legend()
plt.show()