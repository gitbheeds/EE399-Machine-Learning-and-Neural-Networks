#GITHUB: https://github.com/gitbheeds

#import begin
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
#import end



#Problem 1
#github made, available at above address

#Problem 2
#Consider the following data from the class
X=np.arange(0,31)
Y=np.array([30, 35, 33, 32, 34, 37, 39, 38, 36, 36, 37, 39, 42, 45, 45, 41, 40, 39, 42, 44, 47, 49, 50, 49, 46, 48, 50, 53, 55, 54, 53])

#Fit the data to the following model with least squared error
#model of form f(x) = Acos(Bx)+Cx+D
def model (C, x) :
    
    #model 
    f_x =  C[0] * np.cos(C[1]*x) + C[2]*x + C[3]

    return f_x

def loss_func(C, x, y):
    return np.sum((y - model(C, x))**2)

#initial parameter guesses
c0 = np.array([1, np.pi/36, 1, 1])

result = opt.minimize(loss_func, c0, args=(X, Y), method = 'Nelder-Mead')

c_opt = result.x


# Print the optimal parameters
print(c_opt)

# Plot the results
plt.scatter(X, Y, label='Data')
plt.plot(X, model(c_opt, X), 'r-', label='Fit')
plt.legend()
plt.show()

#II Parameter Sweeps

#parameters to sweep through
Aval = np.linspace(0, 10, 100)
Bval = np.linspace(0, 5, 100)
Cval = np.linspace(0, 10, 100)
Dval = np.linspace(0, 10, 20)

#fix B and D, sweep through A and C
loss_grid = np.zeros((len(Bval), len(Dval)))

for i, B in enumerate(Bval):
    for j, D in enumerate(Dval):
        #unreasonably large start
        min_loss = 1212423234

        #sweep A and C
        for A in Aval:
            for C in Cval:
                loss = loss_func([A, B, C, D], X, Y)
                if loss < min_loss : 
                    min_loss = loss
        #store the minimum loss value for the given combination of B and D fixed
        loss_grid[i,j] = min_loss

plt.figure(figsize=(8,6))
plt.pcolor(Dval, Bval, loss_grid, cmap='viridis')
plt.colorbar(label='Loss')
plt.xlabel('D')
plt.ylabel('B')
plt.title('Loss Landscape')
plt.show()