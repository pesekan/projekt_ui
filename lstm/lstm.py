import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Define the function to approximate
def f(x):
    return np.sin(x + np.pi) - 1/(x**2 + 1) + x

# Generate training data
X_train = np.random.uniform(-10, 10, size=(100,)).reshape(-1, 1)
#y_noisy = f(X_train) + np.random.normal(scale=0.1, size=(100,))
#X_train = np.random.uniform(-10, 10, 100).reshape(-1, 1)
y_train = f(X_train) + np.random.normal(scale=0.1, size=(100,))
y_train.reshape(-1, 1)

# Define the LSTM model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.LSTM(units=128, input_shape=(1,1)))
model.add(tf.keras.layers.Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X_train.reshape(-1, 1, 1), y_train, epochs=100, batch_size=32)

# Generate test data
X_test = np.linspace(-10, 10, 20).reshape(-1, 1)

# Make predictions on the test data
y_pred = model.predict(X_test.reshape(-1, 1, 1))

# Plot the results
plt.scatter(X_train, y_train, label='Noisy data')
plt.plot(X_test, y_pred, label='LSTM')
plt.plot(X_test, f(X_test), label='true')
plt.legend()
plt.show()

