# U ovom primjeru smo pokušali dobiti rezultate iz rada Approximation Rate For Neural Networks With General Activation
# Functions, Jonathan W. Siegel i Jinchao Xu, 2021., no za ovaj specifičan slučaj se pokazalo da je znatno bolje
# koristiti veći broj slojeva u neuronskoj mreži nego samo povećati broj neurona u mreži te imati manje slojeva, no
# mogući razlog za to je poprilično malen broj podataka za treniranje zbog čega zapravo i ne nailazimo na problem
# "prokletstvo dimenzionalnosti" te veći broj neurona u manje slojeva nam je donio duže vrijeme izvršavanja te veću
# grešku


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

# Create a simple neural network with one hidden layer
start_time = time.time()

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(256, input_dim=1, activation='relu'))
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='linear'))

# Compile the model
model.compile(loss='mean_squared_error', optimizer='adam')

# Fit the model on the training data
model.fit(x_train, y_train, epochs=1000)

elapsed_time = time.time() - start_time

# Evaluate the model on the test data
test_loss = model.evaluate(x_test, y_test)

# Open file in "append" mode
with open('../vremena.txt', 'a') as file:
    # Write value to end of file
    file.write('Neural network with a single layer and more epochs\n')
    file.write(f'Error: {test_loss}\n')
    file.write(f'Elapsed time: {elapsed_time}\n')

# Plot the original function and the approximated function
plt.scatter(x, y_noisy, label='Noisy data')
plt.plot(x, model.predict(x), color='orange', label='Approximated function')
plt.plot(x, function(x), label='Original function')
plt.legend()
plt.show()