import numpy as np
import matplotlib.pyplot as plt

# Generate random input features and target variable
X = np.random.rand(100, 1)
y = 3*X + 2 + 0.1*np.random.randn(100, 1)

# Define the cost function and its gradient
def cost_function(X, y, theta):
    m = len(y)
    h = np.dot(X, theta)
    error = h - y
    J = np.sum(error**2) / (2*m)
    grad = np.dot(X.T, error) / m
    return J, grad

# Define the gradient descent function
def gradient_descent(X, y, alpha, num_iterations):
    m = len(y)
    n = X.shape[1]
    theta = np.zeros((n, 1))
    J_history = []
    for i in range(num_iterations):
        J, grad = cost_function(X, y, theta)
        J_history.append(J)
        theta = theta - alpha * grad
    return theta, J_history

# Run gradient descent
theta, J_history = gradient_descent(X, y, alpha=0.1, num_iterations=100)

# Plot the input data and the fit
plt.scatter(X, y)
x_fit = np.linspace(0, 1, 100)
y_fit = theta[0] + theta[1]*x_fit
plt.plot(x_fit, y_fit, color='red')
plt.title('Gradient Descent Fit')
plt.show()
