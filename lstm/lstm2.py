# U ovome smo primjeru poboljšali metodu lstm tako što smo povećali broj slojeva neurona u modelu


import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# define the function to approximate
def func(x):
    return np.sin(x + np.pi) - 1/(x**2 + 1) + x

# define the training set
X_train = np.linspace(-10, 10, 100).reshape(-1, 1)
y_train = func(X_train)

# define the test set
X_test = np.linspace(-10, 10, 20).reshape(-1, 1)
y_test = func(X_test)

# create the LSTM model
model = Sequential()
model.add(LSTM(128, input_shape=(1, 1), return_sequences=True))
model.add(LSTM(64, return_sequences=True))
model.add(LSTM(32, return_sequences=True))
model.add(LSTM(16))
model.add(Dense(1))

# compile the model
model.compile(loss='mse', optimizer='adam')

# set early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=10)

# train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stop])

# evaluate the model on the test set
loss = model.evaluate(X_test, y_test)
print('Test loss:', loss)

# make predictions on the test set
y_pred = model.predict(X_test)

# plot the results
plt.scatter(X_train, y_train, label='Noisy data')
plt.plot(X_test, y_test, label='True')
plt.plot(X_test, y_pred, label='Predicted')
plt.legend()
plt.show()
