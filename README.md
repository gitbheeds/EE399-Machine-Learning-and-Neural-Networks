# EE399-Work
Storage for EE399 HW
## Table of Contents (to be added later)


## HW1 Writeup: Curve Fitting with Least Square Error

### Abstract:

This writeup contains the results of curve fitting a dataset with several different models. The merit of these curve fits will 
be determined using the least-squares error method. We will fit 4 separate models to the data. These are a trignometric function
the form `f(x) = Acos(Bx) + Cx + D`, a line, a parabola of the form `Ax^2 + Bx + C`, and a 19th degree polynomial. The first
model will be trained on the entire dataset, and the remaining models will be trained on different parts of the original data
and tested on the remaining parts. We will calculate the Least-Squares error to determine how well these models performed. For 
the trigonometric function, we will also perform several parameter sweeps on the coefficients A, B, C, and D to find minima of the error function.


### Section 1: Overview

The first part of the homework requires us to fit a function in the form `Acos(Bx) + Cx + D` to a given data set. Then, we will
optimize the parameters A, B, C, and D to minimize the Least-Squares error and determine optimal values for the parameters.
With these parameters, we will perform multiple 2D parameter sweeps to find minima in the loss function for a range of parameter
values.

The second part of this homework divides the initial data set into a 20 point training set, and a 10 point test set. A line, 
a parabola, and a 19th degree polynomial will be fit to this data and evaluated based on the Least-Squares error from the test and 
training set. 

The final part of this homework repeats the second part, but using different training and test data from the same initial set. These
results will be compared to those of the previous part.

### Section 2: Background and Important Concepts

This section will cover some important terminology needed to understand the work done in this homework.

#### Least-Squares Error
This is a common statistical technique to fit models to data. The least-squares error is defined as the 
sum of the squared differences between predicted values of the model and the actual values of the data.
Below is the Least-Squares Error equation: 
![Least-Squares Error ](https://user-images.githubusercontent.com/80599571/230671996-23294fdf-f84c-4431-9348-834e1d268cfd.png)

#### Parameter Optimization
Parameter optimization is the process by which the parameters for a model are calculated to minimize a given loss function. There are 
many different loss functions, but we will be using the Least-Squares Error technique as our loss function for this homework. Optimization
requires an initial guess for each parameter, and the quality of the initial guess will improve the optimization. Usually, values are determined
from earlier knowledge of the problem or data. In Python, we can use the `opt.minimize()` function call. 

#### Loss Landscapes
Another way to visualize optimization is to generate a multi-dimensional loss landscape. This is done by fixing a few parameters and sweeping through values of the
remaining parameters. Using a function like `plt.pcolor()` can graphically locate minima and maxima, and can be used to educate initial guesses for parameter optimization.

### Section 3: Development and Implementation of Functions and Algorithms
Note that for all models, the array C is used to store all the coefficients. This was done
to speed up code writing time, and does not have any large impact on the results. 
- Models
  - Trigonometric  
  This model was written using the NumPy library, and contained in a function.
    ![Trig model wrapper function](https://user-images.githubusercontent.com/87102849/231055454-3f269539-8abf-48df-9f74-bf616546f256.png)

  - Line  
  This model was written with basic math operations and contained in a function. 
    ![Line model wrapper function](https://user-images.githubusercontent.com/87102849/231055549-8ca588d4-5438-4b6c-ac23-a7eb4fc4081c.png)

  - Parabola  
  This model was written with basic math operations and contained in a function. 
    ![Parabola wrapper function](https://user-images.githubusercontent.com/87102849/231055603-5c6562fd-90b7-46f7-b31b-37b010e70bda.png)

- Least-Squares Error  
A version of the Least-Squares Error loss function was written for each of the above models. The 19th degree
  polynomial also has a special function, as it contains the optimization through the `np.polyfit()` function.
  ![photo of all the loss functions](https://user-images.githubusercontent.com/87102849/231056066-d5489c47-c69e-4dcb-97c5-bb38f82f58d5.png)

- Minimization of Least-Squares Error  
For all loss functions excluding `loss_func_poly()`, optimization had to be done to minimize the error. This was done
  using the scipy optimize package, specifically `opt.minimize()`. In this function, an optimization type must be
  specified. We used a "Nelder-Mead" optimization scheme. 
  
- Loss Landscapes (Trigonometric Model)
  Loss landscapes were generated using the following method. First, large arrays were made to create parameter sweeps for
  each parameter (Shown below). ![parameter sweeps](https://user-images.githubusercontent.com/87102849/231058301-ccee2288-48f2-44ee-8659-3199d472ae2d.png)  
  These ranges were chosen around the optimized parameters calculated in the first part of the homework. Then, 
  two parameters were chosen to be fixed to the optimized values, and the other two would be swept through using a nested for loop. A matrix was
  instantiated to store the loss function output at each combination of swept parameters. Finally, `plt.pcolor()` was used to
  display the loss landscape. A sample of the loss landscape generation code is below. 
  
  ![loss landscape calc example](https://user-images.githubusercontent.com/87102849/231058133-011e6310-b534-4285-85fe-3408b7de0351.png)

### Section 4: Results
This section contains the results of each task from this homework

#### Trigonometric Model Fitting
The trigonometric model fit the data relatively well, with a total error calculated at 1.59272. Optimized parameters, as well as
a graph of the fit, were printed as output, shown below.  

![trig model results](https://user-images.githubusercontent.com/87102849/231059104-441a6f80-049d-4e09-83e9-95bda2d24d52.png)

#### Loss Landscapes for Trigonometric Model
These pcolor plots help us identify loss function minima for different values of parameters. One might notice that the optimized
parameters, specifically on Loss Landscape, A and C fixed, don't necessarily match the location of the absolute minima. Parameter B
was optimized to be a value near zero, which is a low loss section of the landscape, but it is not an absolute minima

![A B Fixed](https://user-images.githubusercontent.com/87102849/231059332-a4afc53f-3d55-4127-998e-9732b0020545.png)

![A C Fixed](https://user-images.githubusercontent.com/87102849/231059377-09516002-70be-4797-96c2-4bb728ddf177.png)

![A D Fixed](https://user-images.githubusercontent.com/87102849/231059435-5c9071a6-f32b-4f5d-8432-891f4ebfc204.png)

![B C Fixed](https://user-images.githubusercontent.com/87102849/231059464-0d61ef9b-dfd0-4707-909a-67df3da148cf.png)

![B D Fixed](https://user-images.githubusercontent.com/87102849/231059510-c11d2dbb-0063-4509-b5e8-98fe7763c8db.png)

![C D Fixed](https://user-images.githubusercontent.com/87102849/231059571-fda314d9-bc46-4ce6-b5d1-32700b3764a4.png)


#### Evaluation of Other Models using Training/Test Set 1
Training Data Set 1 used the first 20 points of the initial data, and Test Data Set 1 used the last 10 points.  
![TDS1 results](https://user-images.githubusercontent.com/87102849/231061529-c98088c5-b462-4500-9d21-3a349d83be1c.png)  
Note that all models perform better on the training data than the test data. This might be because the training data has more points,
so a tighter fit can be calculated.  

Another important characteristic here is the error on the polynomial fit is nearly zero on training data, and unreasonably large on the test data. 
It's likely that the polynomial fits the training data well, but ends up overfitting the test data due to end behaviors of high degree
polynomials. 

#### Evaluation of Other Models using Training/Test Set 2
Training Data Set 2 used the first and last 10 points of the intial data, and Test Data Set 2 used the 10 points in the middle. 
![TDS2 results](https://user-images.githubusercontent.com/87102849/231062291-c907e787-b65d-483b-93aa-e18294265e28.png)  
This training/test set performed worse on both data sets than set 1. This can be explained by a couple of factors. First, because
the data set has relatively positive correlation, there is a large difference between the first and last 10 datapoints. This means
that curvefitting will require optimizing these far points, causing more error in the midsection of the data, as we can see in the results. 
Interestingly, the polynomial performs better on the test set than the training set, again owing to end behavior of high degree polynomials.


### Section 5: Conclusion
This homework required us to develop and analyze different models for machine learning, and provided good practice into simple optimization techniques.
