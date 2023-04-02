import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Define a nonlinear function to fit
def my_func(x, a, b, c):
    return a * np.exp(-b * x) + c

# Generate some sample data
xdata = np.linspace(0, 4, 50)
ydata = my_func(xdata, 2.5, 1.3, 0.5) + 0.2 * np.random.normal(size=len(xdata))

# Fit the function to the data
popt, pcov = curve_fit(my_func, xdata, ydata)

# Plot the results
plt.scatter(xdata, ydata, label='Data')
plt.plot(xdata, my_func(xdata, *popt), 'r-', label='Fit')
plt.legend()
plt.show()