# Gradientni spust je metoda kojom se često u manjim neuronskim mrežama nalaze parametri modela, dok se u većim mrežama
# umjesto gradientnog spusta često koristi stohastički spust
# U idućem je primjeru dana jednostavna implementacija rada gradijentnog spusta na jednostavnom primjeru

# Define the cost function
def f(x):
    return x**2 + 5

# Define the derivative of the cost function
def df(x):
    return 2*x

# Define the initial point and learning rate
x0 = 5
lr = 0.1

# Perform gradient descent for a fixed number of iterations
for i in range(100):
    # Compute the gradient of the cost function
    grad = df(x0)
    # Update the parameter in the direction of the negative gradient
    x0 -= lr * grad
    # Print the current parameter and cost
    print("x = {:.2f}, f(x) = {:.2f}".format(x0, f(x0)))