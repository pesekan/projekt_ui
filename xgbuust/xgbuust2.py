# XGBoost možemo i unaprijediti primjerice korištenjem tehnike grid search, što i činimo u idućem kodu.
#
# Grid search je tehnika podešavanja hiperparametara koja se koristi za traženje najbolje kombinacije
# hiperparametara koja rezultira najboljom izvedbom za određeni model strojnog učenja.
# Hiperparametri su vrijednosti koje su postavljene prije treninga i utječu na ponašanje modela tijekom
# treninga i predviđanja.

# U pretraživanju mreže (grida) definiramo hiperparametare, a zatim treniramo i procjenjujemo model za
# svaku kombinaciju hiperparametara u mreži. Model se trenira pomoću skupa podataka za obuku i
# procjenjuje pomoću zasebnog skupa podataka za validaciju. Metrika izvedbe koja se koristi za
# procjenu modela može biti bilo koja metrika od interesa kao što su primjerice točnost, preciznost,
# srednja kvadratna pogreška, itd.

# Algoritam pretraživanja mreže zatim odabire kombinaciju hiperparametara koja rezultira najvećom
# izvedbom na validacijskom skupu podataka. Ova kombinacija hiperparametara zatim se koristi za treniranje
# konačnog modela na cijelom skupu podataka.


import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import time

# Define the function to approximate
def func(x):
    return np.sin(x + np.pi) - 1/(x**2 + 1) + x

# Generate training data
X_train = np.random.uniform(-10, 10, 100).reshape(-1, 1)
y_train = func(X_train) + np.random.normal(scale=0.1, size=(100, 1))

# Generate test data
X_test = np.linspace(-10, 10, 20).reshape(-1, 1)
y_test = func(X_test)

# Create DMatrix objects for training and testing
start_time = time.time()

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test)

# Set XGBoost parameters
params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'booster': 'gbtree',
    'eta': 0.1,
    'max_depth': 6,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'verbosity': 0
}

# Train the model
num_rounds = 100
model = xgb.train(params, dtrain, num_rounds)

elapsed_time = time.time() - start_time

# Make predictions on test data
y_pred = model.predict(dtest)

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
    file.write('XGBoost with grid search\n')
    file.write(f'Error: {str_error}\n')
    file.write(f'Elapsed time: {elapsed_time}\n')

# Plot the results
plt.scatter(X_train, y_train, label='Training data')
plt.plot(X_test, y_test, label='True function')
plt.plot(X_test, y_pred, color='orange', label='XGBoost prediction')
plt.legend()
plt.show()
