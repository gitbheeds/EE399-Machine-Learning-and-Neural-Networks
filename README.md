# EE399-Work
A repository containing work on ML/AI algorithms. Learning how to use ML/AI to approach various types of problems. 

## Table of Contents
- [HW 1 Writeup: Curve Fitting with Least Square Error](https://github.com/gitbheeds/EE399-Work/blob/main/README.md#hw1-writeup-curve-fitting-with-least-square-error)
  - [Abstract](https://github.com/gitbheeds/EE399-Work/blob/main/README.md#abstract)
  - [Overview](https://github.com/gitbheeds/EE399-Work/blob/main/README.md#section-1-overview)
  - [Background and Important Concepts](https://github.com/gitbheeds/EE399-Work/blob/main/README.md#section-2-background-and-important-concepts)
  - [Development and Implementation of Functions and Algorithms](https://github.com/gitbheeds/EE399-Work/blob/main/README.md#section-3-development-and-implementation-of-functions-and-algorithms)
  - [Results](https://github.com/gitbheeds/EE399-Work/blob/main/README.md#section-4-results)
  - [Conclusion](https://github.com/gitbheeds/EE399-Work/blob/main/README.md#section-5-conclusion)
- [HW2 Writeup: Correlation Matricies and SVD](https://github.com/gitbheeds/EE399-Work#hw2-writeup-correlation-matricies-and-svd)
  - [Abstract](https://github.com/gitbheeds/EE399-Work#abstract-1)
  - [Overview](https://github.com/gitbheeds/EE399-Work#section-1-overview-1)
  - [Background and Important Concepts](https://github.com/gitbheeds/EE399-Work#section-2-background-and-important-concepts-1)
  - [Development and Implementation of Algorithms](https://github.com/gitbheeds/EE399-Work#section-3-development-and-implementation-of-algorithms)
  - [Results](https://github.com/gitbheeds/EE399-Work#section-4-results-1)
  - [Conclusion](https://github.com/gitbheeds/EE399-Work#section-5-conclusion-1)
- [HW3 Writeup: Classification of Digits in the MNIST_784 Dataset Using Several Supervised Machine Learning Algorithms](https://github.com/gitbheeds/EE399-Work/blob/main/README.md#hw3-writeup-classification-of-digits-in-the-mnist_784-dataset-using-several-supervised-machine-learning-algorithms)  
  - [Abstract](https://github.com/gitbheeds/EE399-Work/blob/main/README.md#abstract-2)  
  - [Overview](https://github.com/gitbheeds/EE399-Work/blob/main/README.md#section-1-overview-2)  
  - [Background and Important Concepts](https://github.com/gitbheeds/EE399-Work/blob/main/README.md#section-2-background-and-important-concepts-2)  
  - [Development and Implementation of Functions and Algorithms](https://github.com/gitbheeds/EE399-Work/blob/main/README.md#section-3-development-and-implementation-of-functions-and-algorithms-1)  
  - [Results](https://github.com/gitbheeds/EE399-Work/blob/main/README.md#section-4-results-2)  
  - [Conclusion](https://github.com/gitbheeds/EE399-Work/blob/main/README.md#section-5-conclusion-2)
- [HW4 Writeup: Training Neural Nets](https://github.com/gitbheeds/EE399-Work/blob/main/README.md#hw-4-training-neural-nets)
  - [Abstract](https://github.com/gitbheeds/EE399-Work/blob/main/README.md#abstract-3)
  - [Overview](https://github.com/gitbheeds/EE399-Work/blob/main/README.md#section-1-overview-3)
  - [Background and Important Concepts](https://github.com/gitbheeds/EE399-Work/blob/main/README.md#section-2-background-and-important-concepts-3)
  - [Development and Implementation of Algorithms](https://github.com/gitbheeds/EE399-Work/blob/main/README.md#section-3-development-and-implementation-of-algorithms-1)
  - [Results](https://github.com/gitbheeds/EE399-Work/blob/main/README.md#section-4-results-3)
  - [Conclusion](https://github.com/gitbheeds/EE399-Work/blob/main/README.md#section-5-conclusion-3)
- [HW5 Writeup: Forecasting Dynamics of the Lorenz Equation](https://github.com/gitbheeds/EE399-Work#hw-5-forecasting-dynamics-of-the-lorenz-equations)
  - [Abstract](https://github.com/gitbheeds/EE399-Work/blob/main/README.md#abstract-4)
  - [Overview](https://github.com/gitbheeds/EE399-Work/blob/main/README.md#section-1-overview-4)
  - [Background and Important Concepts](https://github.com/gitbheeds/EE399-Work/blob/main/README.md#section-2-background-and-important-concepts-4)
  - [Development and Implementation of ML Architectures and Other Functions](https://github.com/gitbheeds/EE399-Work/blob/main/README.md#section-3-development-and-implementation-of-ml-architectures-and-other-functions)
  - [Results](https://github.com/gitbheeds/EE399-Work/blob/main/README.md#section-4-results-4)
  - [Conclusion](https://github.com/gitbheeds/EE399-Work/blob/main/README.md#section-5-conclusion-4)  
- [HW6 Writeup: Analysis of Sea Temperature Data Using PyShred](https://github.com/gitbheeds/EE399-Work/blob/main/README.md#hw6-writeup-analysis-of-sea-temperature-data-using-pyshred)
  - [Abstract](https://github.com/gitbheeds/EE399-Work/blob/main/README.md#abstract-5)
  - [Overview](https://github.com/gitbheeds/EE399-Work/blob/main/README.md#section-1-overview-5)
  - [Background and Important Concepts](https://github.com/gitbheeds/EE399-Work/blob/main/README.md#section-2-background-and-important-concepts-5)
  - [Development and Implementation of Algorithms](https://github.com/gitbheeds/EE399-Work/blob/main/README.md#section-3-development-and-implementation-of-algorithms-2)
  - [Results](https://github.com/gitbheeds/EE399-Work/blob/main/README.md#section-4-results-5)
  - [Conclusion](https://github.com/gitbheeds/EE399-Work/blob/main/README.md#section-5-conclusion-5)  
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
[Back to Table of Contents](https://github.com/gitbheeds/EE399-Work#table-of-contents)




## HW2 Writeup: Correlation Matricies and SVD

### Abstract:  
This homework focused on the yalefaces data set, containing 9 different faces with about 65 lighting scenes for each face (2414 faces in all). We analyzed this data using both correlation matricies and SVD. We compute correlation matricies for small samples of the data set, 100 faces and then 10 faces. We will then compare the results of the correlation matrix of the entire data set against the SVD for the entire data set. We will use the norm of difference between the normalized eigenvectors of the correlation matrix and the first 6 principal component directions of the SVD. We will also plot the first 6 SVD modes and compare their percent variance from the original data. 

### Section 1: Overview
First, we will compute a correlation matrix for the first 100 images in the yalefaces dataset, and plot the matrix with `plt.pcolor()`. Using this matrix, we will then plot the faces that are most correlated and the faces that are the least correlated.  
Then, we will compute another correlation matrix, but this time for 10 images throughout the data set. We will use images 1, 313, 512, 5, 2400, 113, 1024, 87, 314, 2005. We will then plot this correlation matrix in the same fashion as above.  
Next, we will turn our attention to the entire yalefaces data set. We'll create a correlation matrix for the dataset, and then find the six eigenvectors with the largest magnitude eigenvalues. We'll then normalize the eigenvectors so we can compare it against the SVD.  
We will then perform SVD on the dataset and find the first six principal component directions.   
Now, we can compare the first eigenvector with the first prinicipal component direction using the norm of difference.  
Finally, we will compute the percentage of variance of each of the 6 SVD modes to the original data set. 

### Section 2: Background and Important Concepts

#### Correlation Matrix: 
A correlation matrix is used to see how well a dataset maps to another parameter. In this case, we are comparing yalefaces to itself to see which faces are the most similar. There are many metrics to use to determine correlation, but in this case we use a dot product to compare how each face correlates to other faces. Correlation matrices are commonly used in data analysis and machine learning to identify patterns and relationships between variables. Making a heatmap from a correlation matrix is a good way of visualizing the data.  

#### Eigenvectors and Eigenvalues:
An eigenvector is a nonzero vector that, when multiplied by a matrix, results in a scalar multiple of itself. That is, if A is a matrix and v is an eigenvector of A, then Av = λv, where λ is a scalar known as the eigenvalue associated with v. In other words, the action of A on v is simply to scale v by a certain amount λ.

#### Singular Value Decomposition:  
Singular Value Decomposition (SVD) is a matrix factorization technique that decomposes a matrix into three matrices: U, Σ, and V. The U matrix contains the left singular vectors, which represent the contribution of each original variable to the new feature space. The Σ matrix contains the singular values, which represent the strength of each feature in the new feature space. The V matrix contains the right singular vectors, which represent the contribution of each observation to the new feature space. Together, the three matrices allow for the identification of the most important features in a dataset and the reduction of the dimensionality of the dataset without losing important information. SVD has a wide range of applications in data analysis and machine learning, including dimensionality reduction, data compression, and noise reduction.  

The modes of SVD refer to the three matrices obtained during the SVD of a matrix. The left singular vectors (U) represent the original variables, the singular values (Σ) represent the strength of each feature, and the right singular vectors (V) represent the observations. The modes are used to transform the original matrix into a new feature space, where each feature is a linear combination of the original variables, and each observation is a linear combination of the new features. The first column of U and V and the first singular value represent the most important feature, and they are ordered by decreasing importance. The SVD is a powerful tool for data analysis and machine learning because it allows for the identification of the most important features in a dataset and the reduction of the dimensionality of the dataset without losing important information.

### Section 3: Development and Implementation of Algorithms

#### Correlation Matrix:
Correlation matricies were implemented using `np.dot()` We calculated the dot product of a sample set of yalefaces and its transpose. This lets us compare faces to each other in the data set. The matrix was then displayed with a heatmap, generated using `plt.pcolor()` An implementation is shown below:  
![cor_mat example](https://user-images.githubusercontent.com/87102849/232674536-e108d494-c932-4b20-8690-2fae848382a7.png)  

#### SVD: 
The SVD was simply implemented using the numpy linear algebra package. The modes of the SVD were then extracted from the transpose of V. An example is shown below:  

![svd-implementation](https://user-images.githubusercontent.com/87102849/232675018-6589e8fc-90eb-4354-b808-87b31d6a471b.png)


### Section 4: Results

First, a correlation matrix was created for the first 100 images in yalefaces. The heatmap of this matrix is shown below:  

![correlation matrix](https://user-images.githubusercontent.com/87102849/232675541-1fa92018-4340-4c42-bf8d-282f67c1e83c.png)  

Then, we found the most and least correlated images by finding the highest and lowest values in the correlation matrix, taking note of what row and column indices were tagged. These images were then plotted side by side, as shown below:  

![face correlations](https://user-images.githubusercontent.com/87102849/232676163-a1486556-3609-4945-8bff-4b68918ed365.png)  

Next, we narrowed our focus to 10 specific images in the data set, and created a small correlation matrix for this sample, the heatmap of which is shown below:  
![10x10 correlation matrix](https://user-images.githubusercontent.com/87102849/232676395-fe0d2413-1e05-423d-9878-dfd640a684a3.png)

For the final portion of the assignment, we turned our attention to the full dataset, first finding a correlation matrix for it, then calculating the eigenvectors and eigenvalues of the matrix. The eigenvectors were then sorted in descending order of the magnitude of their eigenvalues. These were normalized, and printed to the console:  
![eigvecs](https://user-images.githubusercontent.com/87102849/232677836-1fd53524-306c-44b2-8112-a85870e879b0.png)  

These will be compared to the first six SVD principal component directions (modes), using a norm of difference. The 6 SVD principal component directions were found as follows:  

![pcds](https://user-images.githubusercontent.com/87102849/232678041-e1f68083-afe2-4b66-97a2-956de3dfffee.png)  

The norm of difference was calculated between the first eigenvector and the first principal component direction and is shown below. Note that the difference is neglibible.  
![n_o_d](https://user-images.githubusercontent.com/87102849/232678234-237e8998-32c0-4654-b9bf-58df606eb672.png)  

Finally, we computed the percentage of variance of the 6 SVD modes to the dataset. This was done by comparing the sum of squares of projections onto each SVD mode against the sum of squares of the original dataset. The 6 modes are pictured below, indicating each mode's variance from the dataset. We can see that each successive mode is more and more accurate, with less percent variance.  
![SVD faces](https://user-images.githubusercontent.com/87102849/232678622-2e4d6ecb-1775-42ad-96c0-f6eeb33264f0.png)

### Section 5: Conclusion
This homework focused on understanding the SVD, a powerful, easy to implement technique, and seeing how it compares to other methods of analysis. SVD will definitely be even better in higher dimension problems.  


[Back to Table of Contents](https://github.com/gitbheeds/EE399-Work#table-of-contents)  

## HW3 Writeup: Classification of Digits in the MNIST_784 Dataset Using Several Supervised Machine Learning Algorithms

### Abstract: 

In this assignment, we are analyzing the MNIST_784 dataset. This data set consists of 70,000 labeled images of handwritten digits. We will take a look at the SVD of this data, and analyze it in PCA space. We will use three different supervised learning algorithms to do this: Linear classification, support vector machines, and decision trees. The latter two techniques are highly regarded as good machine learning algorithms, so we will compare their performance to linear classifiers in the context of separating all 10 digits. 

### Section 1: Overview

First, we will load in the MNIST dataset, preprocess it, and then compute the singular value decomposition. We can then determine how many principal component directions(PCD) are required to reconstruct the images with at least 90% confidence. This can be done by analyzing the singular value spectrum, which contains information on how the principal components are weighed in image reconstruction.  

Next, we will visualize the data by graphing the MNIST dataset projected onto 3 principal component directions, which allows us to see PCD contributions to each type of digit in the dataset. This will help us start the next task, which involves using a linear classifier to separate 2 digits in the dataset. We will then increase that to separating three digits to see how the performance of the classifier changes. To further analyze the classifier's capability, we will run it through all combinations of two digits possible to see which two digits are the easiest and hardest to separate.  

Finally, we can compare the linear classfier to the support vector machine and the decision tree. For this performance test, we are determining which of the three classifiers can separate all ten digits the best. These results will be discussed in section 4.  

### Section 2: Background and Important Concepts

#### MNIST_784:
The MNIST dataset, as mentioned in the abstract, is a dataset consisting of 70,000 greyscale images of handwritten digits from zero through nine. Each digit is labeled, and of resolution 28x28. The data is stored in a 70000 x 784 matrix, where each image's pixels is stored as each entry across the row. Each pixel is assigned a value from 0 to 255.0. To preproccess this data, we will do two things. First, we will normalize the data to a range of 0 to 1. Second, we will transpose the matrix so that each image is a column vector, with each entry down the column corresponding to a pixel in the image.  

#### Singular Value Decomposition, **U**, **&Sigma;**, and **V** Matricies:
Singular value decomposition (SVD) is a mathematical technique used to decompose a matrix into three constituent parts, with the aim of simplifying and better understanding the data represented by the matrix. SVD is a powerful tool that is widely used in machine learning, data analysis, and signal processing applications. The three matricies generated by this technique can be interpreted as follows:  

   - The **U** matrix contains images in the dataset. Each column of **U** is a different image, and each entry down the column represents the contribution of each principal component to that image.  
   - The **&Sigma;** matrix is a diagonal matrix indicating the weights of each principal component. It is arranged in descending order. So in this case, the values in the **&Sigma;** matrix tell us how important each principal component is to all the images in the set. Plotting the values of the singular **&Sigma;** matrix can help one understand how each prinicpal component contributes to image reconstruction.  
   - The **V** matrix is structured such that each row is a principal component, and each entry across the row represents how each pixel contributes to a given principal component.  

#### Linear Classifier:
A linear classifier is a type of machine learning algorithm used for binary or multi-class classification tasks. It is a simple and widely used approach that works by finding a linear boundary or hyperplane in the feature space that separates the different classes of data.

In a binary classification problem, a linear classifier would draw a line to separate the two classes, while in a multi-class problem, it would draw a hyperplane that separates the data into different regions for each class. The location of the boundary or hyperplane is determined during the training phase by optimizing a specific objective function, such as maximizing the margin between the classes or minimizing the classification error.

##### Linear Discriminant Analysis: 
Linear discriminant analysis (LDA) is a popular technique used in linear classification to find the best linear boundary that separates different classes of data. LDA is a supervised learning algorithm that works by projecting the data onto a lower-dimensional space while maximizing the separability between the classes.

The goal of LDA is to find a set of features, also known as discriminants, that can effectively separate the classes. The discriminants are obtained by maximizing a criterion function, such as Fisher's linear discriminant, that captures the differences between the means of the classes while minimizing the within-class scatter.

Once the discriminants are obtained, new, unseen data can be projected onto the same lower-dimensional space, and the class label can be assigned based on which side of the decision boundary it falls on.

One of the strengths of LDA is that it is a simple and efficient algorithm that can work well even when the number of features is larger than the number of samples. LDA also has the ability to handle more than two classes, making it useful for multi-class classification problems.

However, LDA assumes that the data follows a Gaussian distribution and that the classes have equal covariance matrices. In practice, these assumptions may not always hold, and other techniques, such as support vector machines, may be more suitable. Once the linear classifier is trained, it can be used to predict the class of new, unseen data by evaluating which side of the boundary or hyperplane it falls on. Linear classifiers can be fast and efficient to train and can work well in low-dimensional feature spaces. However, they may not be effective in more complex data spaces or when the classes are not linearly separable. In these cases, non-linear classifiers such as neural networks or decision trees may be more appropriate. This is something that will be covered in the results section. 


#### Support Vector Machine:  
Support Vector Machines (SVMs) are a popular and powerful class of supervised machine learning algorithms that can be used for both classification and regression tasks. SVMs work by finding the best possible boundary or hyperplane that separates different classes of data in a high-dimensional feature space.

In a binary classification problem, SVMs aim to find a hyperplane that maximizes the margin between the two classes, i.e., the distance between the hyperplane and the closest data points of each class. This hyperplane is then used to classify new, unseen data based on which side of the boundary it falls on. SVMs are particularly useful in cases where the data is not linearly separable by transforming the original feature space into a higher-dimensional space where a hyperplane can be found.

The process of finding the best hyperplane is done by solving a mathematical optimization problem that involves maximizing the margin while also minimizing the classification error. The SVM algorithm achieves this by using a subset of the training data, known as support vectors, to determine the hyperplane.

One of the strengths of SVMs is that they can work well even in cases where there is noise in the data or when the classes are not well-separated. SVMs also have the ability to handle both linear and nonlinear data using a kernel trick, which maps the data into a higher-dimensional space, where the classes may become separable.

SVMs are widely used in various applications, including text classification, image classification, and bioinformatics. However, SVMs can be computationally intensive, especially for large datasets with many features, and the choice of kernel and hyperparameters can significantly impact the performance of the algorithm.  

#### Decision Tree: 
Decision trees are a type of supervised machine learning algorithm used for classification and regression tasks. Decision trees work by recursively partitioning the feature space into smaller and smaller regions, based on the values of the input features, until each region contains a single class or a regression value.

The process of building a decision tree involves selecting the best feature at each node that can most effectively split the data into subsets with the highest possible purity. The purity of a subset is determined by measuring the homogeneity of the classes or values in that subset. Popular impurity measures include Gini index, entropy, and classification error.

The tree is grown recursively by splitting the data at each node based on the selected feature until a stopping criterion is met, such as reaching a maximum depth or minimum number of data points in a leaf node. Once the decision tree is trained, it can be used to predict the class or regression value of new, unseen data by traversing the tree from the root node to a leaf node.

One of the strengths of decision trees is that they are easy to interpret and can be visualized, making them useful for exploratory data analysis and understanding the relationship between the input features and the target variable. Decision trees are also robust to noisy data and can handle both categorical and numerical features.

However, decision trees can suffer from overfitting, where the tree becomes too complex and fits the training data too well, resulting in poor performance on new data. Various techniques, such as pruning, regularization, and ensemble methods, can be used to address this issue.

### Section 3: Development and Implementation of Functions and Algorithms:

Most of the algorithms are implemented using `sklearn` library in Python. This library is full of useful features for machine learning algorithms. We will also be using this library for generating training and test data, as well as calculating accuracy, and fitting algorithms. Implementation of the SVD was done using `np.linalg.svd` Full matricies were disabled in order to save memory. Otherwise the SVD matricies require too large a memory allocation to run on most personal machines. To determine the number of modes needed to reconstruct the images from the dataset with a given fidelity threshold, a sum of squares average was computed to determine how many modes were needed. A threshold of 90% fidelity was deemed as sufficient reconstruction.   

- `train_test_split()`: This function, provided by `sklearn.model_selection`, takes in arrays and generates training and test sets from these arrays. The `test_size` parameter determines what percentage of the data will be reserved for testing data. The `random_state` parameter selects a random seed, and picks training and test data based on this seed. This ensures psuedo-randomized data in both test and train data.  
- `LinearDiscriminantAnalysis()`: This function provided by `sklearn.discriminant_analysis` handles all the training and accuracy scoring of the linear classifier. This model, as it is named, uses LDA to compute the classification.  
- `SVC()`: This function from `sklearn.svm` creates a Support Vector Classifier, and handles all the training and accuracy scoring of the SVC. 
- `DecisionTreeClassifier()`: This function provided by `sklearn.tree` generates a Decision Tree Classifier, and contains methods for training and scoring the tree.  

A sample of a classifier instantiation is shown below.  
![sample code](https://user-images.githubusercontent.com/87102849/233878214-e4ac709b-a0a9-4ad7-8c21-37d254894f03.png)  

### Section 4: Results
#### Singular Value Spectrum  
The singular value spectrum for the MNIST dataset can be seen below. It is apparent that there is a steep dropoff in the contribuition of a given principal component to the image in the dataset. While this value is hard to see in the graph, the threshold for 90% image reconstruction is calculated and listed below the graph. This point is just after the elbow in the graph.  
![singular value spectrum](https://user-images.githubusercontent.com/87102849/233879105-be7c985a-c986-4361-9471-75347b4451f1.png)  

#### Projection of MNIST onto Three V-modes:
The MNIST data was projected onto **V**-modes 2, 3, and 5. These are relatively important modes, as indicated by the Singular Value Spectrum. The data was colored by digit label, and graphed onto a 3D subspace. Here, we can see how each mode contributes to each digit in the dataset. Note that the hole in the middle of the dataset corresponds to approximately zero contribution of the mode to a digit. This is hard to imagine for 3 highly relevant modes. Different versions of Python apparently graph the same data in a different order, so running the provided code may result in a moderately different shape.  
![projection of MNIST](https://user-images.githubusercontent.com/87102849/233879675-8ddc9844-af27-4466-9a7e-5345a5134380.png)  

#### Linear Classifier for Digits 1 and 5  
The linear classifier was trained to separate digits 1 and 5. The values were first binned, and then separated using `train_test_split()`. Accuracies were reported as follows.  
- Training Set Accuracy: 99.15%  
- Test Set Accuracy: 98.80%

As we can see, the linear classifier did very well separating these two digits.  

#### Linear Classifier for Digits 1, 5, and 7
To increase the workload on the classifier, it was then asked to separate digits 1, 5, and 7. 7 was chosen since it commonly can bear some resemblance to the number 1 in most handwriting styles. As expected, the accuracy of the classifier decreased.  
- Training Set Accuracy: 98.17%
- Test Set Accuracy: 97.37%

#### Linear Classifier: Comparing Accuracy for all Digit Combinations  
Now, we compare which two digits are hardest and easiest to separate.  
- Maximum Test Accuracy: 99.61% separating digits 6 and 7
- Minimum Test Accuracy: 95.02% separating digits 3 and 5  

These results were a little surprising, as we did not expect 3 and 5 to be the hardest to separate, since they have different curvatures. Regardless, the classifier still performed quite well when separating two digits. Next we will try separating all ten digits, and compare its performance to the two other supervised learning algorithms.  

#### Linear Classifier Compared to SVM and Decision Tree  

Using the Linear Classifier for separating all 10 digits performed worse than both of the other two algorithms. The results for the test data are as follows:  
- SVM Accuracy: 98.02%
- Decision Tree Accuracy: 87.40%
- Linear Classifier: 86.69%  

### Section 5: Conclusion
There is a good amount of difference in the accuracy of each algorithm. We can first note that teh SVM grossly outperformed both the decision tree and the linear classifier. Considering the limitations of the other models, this makes some sense. The linear classifier struggles with higher dimensional data that is not in a Gaussian distribution. This is not always an assumption that can be made, indicating that the linear classifier may struggle with oddly distributed data. Also, there is the issue on dimensionality to tackle. The linear classifier is good and fast to use in lower dimensions, as we saw it was highly capable of separating 2 or 3 digits in the dataset. As we increase the dimensionality of the data, it becomes harder to separate the data using hyperplanes.  

In the case of the MNIST dataset, SVMs are often able to achieve higher accuracy than decision trees because they are well-suited to handle high-dimensional data with a large number of features. Each pixel in the MNIST images represents a feature, so there are a total of 784 features. SVMs are designed to find the best possible boundary between the classes, and they can handle non-linearly separable data by using a kernel function.

In contrast, decision trees may not perform as well on the MNIST dataset because they tend to work better with fewer features and may struggle to capture the complex relationships between the pixels in the images. Decision trees also tend to be more prone to overfitting, which can lead to poor generalization performance on new, unseen data. The MNIST dataset, with 784 features, is difficult for a decision tree to separate. 

[Back to Table of Contents](https://github.com/gitbheeds/EE399-Work#table-of-contents)  


## HW 4: Training Neural Nets  

### Abstract 

In this assignment, we will be training a couple neural nets on datasets used in HW 1 and HW 3. Our goal is to analyze how a three layer feed forward neural net (FFNN) compares to the models used in the previous assignments. For the HW 1 dataset, we will compare the loss functions of the three layer FFNN to those of the line, parabola, and polynomial fit computed in the first assignment. This will help us understand how a neural net performs in low dimensionality problems. From there, we will train a similar FFNN on the MNIST_784 dataset used in HW 3. This dataset will test the FFNN's ability to handle much higher dimension problems. As we will see, the neural net does not provide the best results on the simpler, two dimensional data, but outperforms old industry standards in higher dimensions. 

### Section 1: Overview

This assignment can largely be broken into two parts. First, we will train a three layer FFNN on the same dataset from HW 1. Much like in that assignment, we will use two separate training and test data splits, and calculate the loss as a least squares error. The first training set will be the first 20 points of the data, and the remaining 10 points will comprise the test set. For the second training set, we will use the first and last 10 points of the data, and the middle 10 for the test set.  We can then compare the results using the FFNN to the three models we had tested in HW 1.  

The second part of this assignment involves the MNIST_784 dataset. We will compute the PCA of the data, and then train a three layer FFNN on the data to separate all 10 digits. We'll compare the accuracy of the FFNN to a different type of neural net, a Long Short Term Memory Network, as well as the old standards, the Support Vector Machine and the Decision Tree. We can then spend some time comparing the accuracies of these results against the overall runtime of each process to determine where neural nets are the most optimal solution, and where the performance of the neural net is not worth the runtime and overall complexity. 

### Section 2: Background and Important Concepts  

#### HW 1 Dataset
The dataset from HW1 is a simple two-dimensional dataset with 30 total points. Below is an image of the dataset.  
![HW1 dataset](https://user-images.githubusercontent.com/87102849/236311751-318820c4-a094-4790-a8f2-cd98aabdf7e3.png)

#### MNIST_784
The MNIST dataset is a dataset consisting of 70,000 greyscale images of handwritten digits from zero through nine. Each digit is labeled, and of resolution 28x28. The data is stored in a 70000 x 784 matrix, where each image's pixels is stored as each entry across the row. Each pixel is assigned a value from 0 to 255.0. To preproccess this data, we will do two things. First, we will normalize the data to a range of 0 to 1. Second, we will transpose the matrix so that each image is a column vector, with each entry down the column corresponding to a pixel in the image. This allows us to perform PCA on the data. 

#### Least-Squares Error
This is a common statistical technique to fit models to data. The least-squares error is defined as the 
sum of the squared differences between predicted values of the model and the actual values of the data.
Below is the Least-Squares Error equation: 
![Least-Squares Error ](https://user-images.githubusercontent.com/80599571/230671996-23294fdf-f84c-4431-9348-834e1d268cfd.png)

#### Loss Results of Models From HW 1
For reference, the reported loss from the three models tested in HW 1 are provided here. We will again reference these in the results section.  
![HW1 loss results](https://user-images.githubusercontent.com/87102849/236313647-10de1555-c712-4d1c-965a-3b11e543c9f3.png)


#### Feed Forward Neural Net (FFNN)
A feedforward neural network is a type of artificial neural network where the information flows in one direction, forward, from the input layer, through the hidden layers, and to the output layer. In a feedforward neural network, there are no feedback connections between the layers, which means that the output of one layer only depends on the input from the previous layer.

The input layer receives the input data, which is usually a vector of numerical values. The hidden layers process this input and apply non-linear transformations to it. Each hidden layer is composed of a set of neurons or units, and each unit takes a weighted sum of the inputs and applies an activation function to the result. The weights and biases of the neurons are learned during the training phase using a process called backpropagation.

The output layer produces the final output of the network, which could be a classification label, a numerical value, or a probability distribution over possible outputs. The output layer also applies an activation function to the output of the previous layer.

#### Activation Functions (ReLU and Linear)
Activation functions are a critical component of artificial neural networks, as they introduce non-linearity to the output of each neuron. In a neural network, activation functions are applied to the weighted sum of inputs of each neuron to introduce non-linearities in the output. Rectified Linear Unit (ReLU) and Linear activation functions are two commonly used activation functions in neural networks. These are the ones we will be using in our FFNNs. A ReLU activation function returns the input if it is positive and returns zero if it is negative. Mathematically, ReLU can be defined as `f(x) = max(0, x)`. This function is commonly used in hidden layers of neural networks because it allows the model to learn non-linear patterns in the data, while also being computationally efficient. A linear activation function returns the input as it is without any non-linear transformation. Mathematically, linear activation function can be defined as `f(x) = x`. This function is often used in the output layer of regression problems where the goal is to predict a continuous value.

#### Long Short Term Memory Networks (LSTM)
LSTMs are a type of recurrent neural network (RNN) that can process sequential data by preserving information over time. They are different from FFNNs in several ways.

In a feedforward neural network, the information flows in one direction, from the input layer through the hidden layers to the output layer. However, in an LSTM, the information can flow in multiple directions, as the output of the current time step can affect the input of the next time step. This means that LSTM layers generate feedback which can further be used to strengthen the LSTM. LSTMs have a unique architecture that allows them to capture long-term dependencies in the input sequence. They use a special memory cell that can maintain information over time and gates that control the flow of information into and out of the memory cell. The gates are composed of sigmoid activation functions that can decide which information to keep, forget or update in the memory cell. The architecture of an LSTM allows it to handle vanishing and exploding gradients that can occur in RNNs when training on long sequences. The gates can control the amount of gradient that flows through the network, which can prevent the gradient from vanishing or exploding. This specialized memory architecture allows LSTMs to better preserve long-term dependencies in the input data.  

#### Support Vector Machine (SVM):  
Support Vector Machines (SVMs) are a popular and powerful class of supervised machine learning algorithms that can be used for both classification and regression tasks. SVMs work by finding the best possible boundary or hyperplane that separates different classes of data in a high-dimensional feature space. In a binary classification problem, SVMs aim to find a hyperplane that maximizes the margin between the two classes, i.e., the distance between the hyperplane and the closest data points of each class. This hyperplane is then used to classify new, unseen data based on which side of the boundary it falls on. SVMs are particularly useful in cases where the data is not linearly separable by transforming the original feature space into a higher-dimensional space where a hyperplane can be found. The process of finding the best hyperplane is done by solving a mathematical optimization problem that involves maximizing the margin while also minimizing the classification error. The SVM algorithm achieves this by using a subset of the training data, known as support vectors, to determine the hyperplane.

One of the strengths of SVMs is that they can work well even in cases where there is noise in the data or when the classes are not well-separated. SVMs also have the ability to handle both linear and nonlinear data using a kernel trick, which maps the data into a higher-dimensional space, where the classes may become separable. SVMs are widely used in various applications, including text classification, image classification, and bioinformatics. However, SVMs can be computationally intensive, especially for large datasets with many features, and the choice of kernel and hyperparameters can significantly impact the performance of the algorithm.  

#### Decision Tree: 
Decision trees are a type of supervised machine learning algorithm used for classification and regression tasks. Decision trees work by recursively partitioning the feature space into smaller and smaller regions, based on the values of the input features, until each region contains a single class or a regression value. The process of building a decision tree involves selecting the best feature at each node that can most effectively split the data into subsets with the highest possible purity. The purity of a subset is determined by measuring the homogeneity of the classes or values in that subset. Popular impurity measures include Gini index, entropy, and classification error. The tree is grown recursively by splitting the data at each node based on the selected feature until a stopping criterion is met, such as reaching a maximum depth or minimum number of data points in a leaf node. Once the decision tree is trained, it can be used to predict the class or regression value of new, unseen data by traversing the tree from the root node to a leaf node. One of the strengths of decision trees is that they are easy to interpret and can be visualized, making them useful for exploratory data analysis and understanding the relationship between the input features and the target variable. Decision trees are also robust to noisy data and can handle both categorical and numerical features. However, decision trees can suffer from overfitting, where the tree becomes too complex and fits the training data too well, resulting in poor performance on new data. Various techniques, such as pruning, regularization, and ensemble methods, can be used to address this issue.

### Section 3: Development and Implementation of Algorithms

As in the previous assignments, the SVM and decision tree were implemented using the `sklearn` library, as it provides an excellent framework for writing classifiers and generating accuracy scores. An implementation of an SVM using `sklearn` is shown below. The optional parameters for the kernel and random state allow for reproducibility of results.  
![SVM implementation](https://user-images.githubusercontent.com/87102849/236316645-6b85b788-af48-4b47-ad9d-d4af850abf2d.png)  
The implementation of the decision tree is also similar. Again, the random state parameter is set to ensure results are reproducible.  
![DT implementation](https://user-images.githubusercontent.com/87102849/236316910-5f5d5f90-c22d-46ae-b7a1-c6b129f1099b.png)  

Both of the neural nets were implemented using the Keras API. While there are not many functional differences between Keras and PyTorch, two popular deep learning libraries, there were a couple of reasons why Keras was chosen over PyTorch. Keras is built on top of TensorFlow, and is focused on making building and training deep learning models as simple as possible, with a focus on ease-of-use and accessibility. On the other hand, PyTorch is a lower-level library that gives developers more control over the details of their models, with a focus on flexibility and customization. For the sake of understanding the high level operation of our design and not spending too much time at the low level details, Keras was chosen to implement our neural nets. The three layer FFNN was implemented as follows, with an arbitary number of neurons per layer. For part 1 of the homework, the FFNN was trained over 10,000 epochs with a batch size of 10. These numbers were also arbitrarily decided, to ensure that the FFNN generated a robust function. On the MNIST dataset, the FFNN was trained over 10 epochs with a batch size of 32. The output layer had 10 neurons in order to generate a method of classifying all 10 digits. 
![FFNN implementation](https://user-images.githubusercontent.com/87102849/236319021-ce81c1e8-2453-4296-b2bf-bdb06b67ff97.png)  
This implementation was for part 1 of the assignment.  

![part 2 FFNN implementation](https://user-images.githubusercontent.com/87102849/236319652-95bd4a33-ada5-4c27-8df9-7b90bf7b0a2c.png)  
This implemenation was for the MNIST dataset in part 2.

The LSTM was also implemented similarly. Keras allows for a very quick implementation of such a network as follows. On the MNIST dataset, the implementation was as follows. Note that the input data had to be reshaped into higher dimensions to be supplied to the LSTM. The LSTM was similarly trained over 10 epochs with a batch size of 128 to account for the higher dimensionality of the input data. 
![LSTM implementation](https://user-images.githubusercontent.com/87102849/236319919-13261d0f-b208-406d-b13e-26b45001891e.png)


### Section 4: Results

The comparison of the FFNN to the models from HW 1 can be seen here. An important conclusion here is that it took the neural net 10,000 epochs to generate a loss function that outperformed only the polynomial fit on the first test dataset. Similar performance can be seen on the second dataset.  

#### From HW1
##### Training Data Set 1 Loss functions 
- Loss function from line fit  2.242749387090776 
- Loss function from parabola fit  2.2427493872531716 
- Loss function from polynomial fit  0.028351970844281614 

##### Test Data Set 1 Loss functions 

 - Loss function from line fit  3.4392167768613793 
 - Loss function from parabola fit  3.4391905006921237 
 - Loss function from polynomial fit  30022321415.023407

##### Training Data Set 2 Loss functions 

 - Loss function from line fit  3.1615109544610585 
 - Loss function from parabola fit  3.1636135157981684 
 - Loss function from polynomial fit  34471695091300.21 

##### Test Data Set 2 Loss functions 

 - Loss function from line fit  33.573133901279796 
 - Loss function from parabola fit  34.011683144193 
 - Loss function from polynomial fit  4027.9343692981083 

#### Neural Net Results: 10000 Epochs, batch size 10

- Training set 1 error =  29.525825250436466
- Test set 1 error =  26.053422135278232
- Training set 2 error =  237.00022363711562
- Test set 2 error =  738.4990082935087  

#### Runtimes: 
- FFNN: 
  - Set 1: 1m 29.3s
  - Set 2: 2m 42.9s
- HW1 Models (Total runtime for all models)
  - Set 1: Less than 1s
  - Set 2: Less than 1s

#### Part II: MNIST

The classification accuracies and runtimes of the four different classifiers are as follows:

- FFNN accuracy: 97.00%
  - Runtime: 20.2s
- LSTM accuracy: 98.56%
  - Runtime: 2m 39.4s
  - Note that this runtime includes reshaping the data, a mandatory step to use an LSTM
- SVM accuracy: 97.92%
  - Runtime: 5m 26s
- Decision Tree accuracy: 87.54%
  - Runtime: 16.8s   

### Section 5: Conclusion  

The results indicate some clear trends in the use of neural nets. First of all, it seems highly unnecessary to use a neural net for low dimension problems like 2D curve fitting. As we saw, the FFNN did not perform well on either train/test set. Even allowing it to train for 10,000 epochs did not increase its performance in a meaningful way. Instead, we saw incredibly long runtimes. In contrast, it took less than a second to run *all* of the 2D models and calculate their loss functions. From this perspective, we can argue that there is no good reason to use a neural net on a low dimension problem. 

Now considering the MNIST dataset, we see something quite different. The FFNN and LSTM performed very well, and matched or even surpassed the accuracy of the SVM. In HW 3, we determined that the SVM was the best method we had at the time to separate the 10 digits in the MNIST dataset. With the introduction of neural nets, we see that this may no longer be the case.   

First, let us consider the LSTM. It had an accuracy of 98.56%, beating out the SVM by roughly 1%. This isn't an astounding performance increase, but what's important here is the runtimes. Even accounting for the time it takes to reshape the data, the LSTM completed all calculations in roughly half the time that the SVM required. This alone is a good reason to consider the LSTM over a SVM for classifying high dimensionality data.  

Next, we turn our attention to the FFNN. It had roughly the same accuracy as the SVM, but a blisteringly fast runtime, taking about 16.3% of the time for a less than 1% drop in accuracy. While it was not faster than the decision tree, it had a 10% higher accuracy, so it is the better choice between those two. The FFNN looks to be the fastest and most reasonably accurate method to classify large datasets that have high dimensionality.  

In conclusion, neural nets are an incredibly powerful tool, but they shine the best with high dimensionality and complex classification requirements, outperforming all industry standards overall. 

[Back to Table of Contents](https://github.com/gitbheeds/EE399-Work#table-of-contents)  


## HW 5 Forecasting Dynamics of the Lorenz Equations

### Abstract: 

In this assignment, we analyze the Lorenz Equations using four different neural net architectures. Each of the four neural nets (Feed Forward, Long Short Term Memory, Echo State, and Recurrent) are trained on a large set of data from the Lorenz Equations that is generated at different values of &rho;, a parameter that controls the chaos of the equation. Each net will then be tested on two different sets of Lorenz data for different values of &rho; than the training. While the neural nets perform slightly differently each time, we see that all of them struggle to forecast the dynamics of the Lorenz Equations, except the Long Short Term Memory and the Echo State networks. 

### Section 1: Overview

In this assignment, we will train various neural nets to advance the solution of the Lorenz Equation from t to t + &Delta;t for &rho; = 10, 28, and 40. The training data will consist of the concatenation of the outputs for all &rho; values. Each neural net will then be tested on its ability to forecast future states of the Lorenz Equations for first &rho; = 17 and then &rho; = 35. We will analyze the results using Mean-Square Error as our loss function.

### Section 2: Background and Important Concepts


#### Lorenz Equations

The Lorenz equations are a set of three non-linear differential equations that were first studied by the meteorologist Edward Lorenz in 1963. These equations are widely known for their chaotic behavior and have become a classic example in the field of chaos theory. They were originally derived as a simplified model for atmospheric convection, but they have found applications in various fields, including physics, mathematics, and engineering.

The Lorenz equations describe the evolution of a three-dimensional dynamical system. The variables involved in the equations represent the state of the system at any given time. The equations themselves are as follows:  

$\frac{{dx}}{{dt}} = \sigma(y - x)$

$\frac{{dy}}{{dt}} = x(\rho - z) - y$

$\frac{{dz}}{{dt}} = xy - \beta z$

The parameters of the Lorenz Equations are explained below.
σ (Sigma): This parameter represents the Prandtl number, which measures the ratio of momentum diffusivity to thermal diffusivity. It determines the rate at which temperature differences are smoothed out by conduction. Increasing σ leads to more efficient mixing and stabilizes the system. If σ is very low, the system can become periodic or even converge to a stable fixed point. On the other hand, high values of σ can result in chaotic behavior.

ρ (Rho): This parameter represents the Rayleigh number, which measures the ratio of the strength of buoyancy forces to viscous forces. It determines the tendency of the system to exhibit convection or circulation. Increasing ρ leads to more vigorous convection and increases the complexity of the system. At low values of ρ, the system can be stable with simple periodic behavior. As ρ increases beyond a critical value, the system becomes chaotic. This critical value is around 28. 

β (Beta): This parameter represents the ratio of the width to the height of the convection cell. It affects the stability and symmetry of the system. Changes in β do not significantly alter the dynamics of the system, but they can affect the shape and structure of the attractor.

It's important to note that the Lorenz equations are highly sensitive to initial conditions. Even small changes in the initial values of x, y, and z can lead to drastically different trajectories. This sensitivity to initial conditions is one of the defining characteristics of chaotic systems.  

As mentioned in the overview, we will only manipulate the value of &rho; to influence the chaos of the system, and not change initial values of x, y, z, &Beta;, or &sigma;.

#### Activation Functions
Activation functions are a critical component of artificial neural networks, as they introduce non-linearity to the output of each neuron. In a neural network, activation functions are applied to the weighted sum of inputs of each neuron to introduce non-linearities in the output. Rectified Linear Unit (ReLU) and Linear activation functions are two commonly used activation functions in neural networks. These are the ones we will be using in our FFNNs. A ReLU activation function returns the input if it is positive and returns zero if it is negative. Mathematically, ReLU can be defined as `f(x) = max(0, x)`. This function is commonly used in hidden layers of neural networks because it allows the model to learn non-linear patterns in the data, while also being computationally efficient. A linear activation function returns the input as it is without any non-linear transformation. Mathematically, linear activation function can be defined as `f(x) = x`. This function is often used in the output layer of regression problems where the goal is to predict a continuous value.  

In this assignment, we regularly use ReLU activation for all layers excluding the output layer, which has a linear activation function. 

#### Loss Function
Loss functions are an important way to classify the performance of a neural net. Using a defined loss function, we are able to determine how accurate a neural net was able to solve the problem it was given. Most commonly, neural nets are configured with a Mean-Square Error loss function, and that is what we will be using in this assignment. 

#### Feed Forward Neural Net (FFNN)

A feedforward neural network is a type of artificial neural network where the information flows in one direction, forward, from the input layer, through the hidden layers, and to the output layer. In a feedforward neural network, there are no feedback connections between the layers, which means that the output of one layer only depends on the input from the previous layer.

The input layer receives the input data, which is usually a vector of numerical values. The hidden layers process this input and apply non-linear transformations to it. Each hidden layer is composed of a set of neurons or units, and each unit takes a weighted sum of the inputs and applies an activation function to the result. The weights and biases of the neurons are learned during the training phase using a process called backpropagation.

The output layer produces the final output of the network, which could be a classification label, a numerical value, or a probability distribution over possible outputs. The output layer also applies an activation function to the output of the previous layer.

In this assignment, our FFNN will be three layers deep, consisting of 3 neurons, 64 neurons, and 3 neurons. 

#### Recurrent Neural Network (RNN)
A Recurrent Neural Network (RNN) is a type of neural network specifically designed to process sequential or time-dependent data. Unlike feedforward neural networks, where information flows only in one direction, RNNs introduce loops within the network, allowing information to persist and be passed from one step to another.

The core idea behind an RNN is that it maintains an internal hidden state that serves as a form of memory. At each time step, the RNN takes an input and combines it with the hidden state from the previous step to produce an output and update the current hidden state. This process enables the network to capture and leverage temporal dependencies in the data.

In terms of architecture, an RNN typically consists of three main components: an input layer, a hidden layer (also known as the recurrent layer), and an output layer. The hidden layer is responsible for maintaining the internal state and processing sequential information. The input and output layers connect to the hidden layer, enabling the network to receive input data and produce predictions or outputs.

Training an RNN involves optimizing its parameters to minimize the discrepancy between the predicted outputs and the ground truth. This is achieved through a process called backpropagation through time (BPTT), which is an extension of the standard backpropagation algorithm used in feedforward neural networks. BPTT calculates the gradients by unfolding the recurrent structure of the network over time and propagating the errors backward through the unfolded structure.

RNNs find applications in various domains, including natural language processing (e.g., language modeling, machine translation), speech recognition, handwriting recognition, and music generation. Their ability to model sequential information makes them well-suited for tasks where the order of the input data matters and where context and dependencies play a crucial role.

However, RNNs can suffer from challenges like the vanishing or exploding gradient problem, which occurs when gradients either diminish or grow exponentially over time, making it difficult to capture long-term dependencies. 


#### Long Short Term Memory Network (LSTM)
LSTMs are a type of recurrent neural network (RNN) that can process sequential data by preserving information over time. They are different from FFNNs in several ways.

In a feedforward neural network, the information flows in one direction, from the input layer through the hidden layers to the output layer. However, in an LSTM, the information can flow in multiple directions, as the output of the current time step can affect the input of the next time step. This means that LSTM layers generate feedback which can further be used to strengthen the LSTM. LSTMs have a unique architecture that allows them to capture long-term dependencies in the input sequence. They use a special memory cell that can maintain information over time and gates that control the flow of information into and out of the memory cell. The gates are composed of sigmoid activation functions that can decide which information to keep, forget or update in the memory cell. The architecture of an LSTM allows it to handle vanishing and exploding gradients that can occur in RNNs when training on long sequences. The gates can control the amount of gradient that flows through the network, which can prevent the gradient from vanishing or exploding. This specialized memory architecture allows LSTMs to better preserve long-term dependencies in the input data.  

#### Echo State Network (ESN)
An Echo State Network (ESN) is a type of recurrent neural network (RNN) that is designed to efficiently process temporal data. It was introduced as a simplified approach to training recurrent neural networks, addressing some of the challenges associated with training traditional RNNs.

The key characteristic that sets ESNs apart from other RNNs is the concept of an "echo state." An echo state refers to a fixed internal reservoir of recurrently connected nodes, which is randomly initialized and remains untrained throughout the learning process. The input data is fed into this reservoir, and the reservoir's dynamics generate a high-dimensional representation of the input sequence. The reservoir essentially acts as a dynamic memory that captures the temporal dependencies in the input data.

The output of an ESN is computed by applying a linear transformation to the reservoir's states. This transformation is learned through a simple training procedure, typically employing linear regression or another shallow learning algorithm. The fixed, untrained nature of the reservoir helps overcome the vanishing/exploding gradient problem that often hinders the training of traditional RNNs.

ESNs have been successfully applied to various time series prediction tasks, such as speech recognition, weather forecasting, and financial market analysis. Their simplicity and computational efficiency make them particularly suitable for real-time applications, where quick and accurate predictions are crucial. Additionally, ESNs have shown good generalization capabilities, even with limited training data, making them useful in scenarios with limited labeled examples.

Overall, the key advantages of ESNs include their easy training procedure, ability to handle complex temporal patterns, and robustness to noise and input variations. However, they may not be as effective as more complex RNN architectures for tasks requiring fine-grained control or precise sequence generation, as their predictions heavily rely on the reservoir dynamics.

### Section 3: Development and Implementation of ML Architectures and Other Functions

Unlike previous assignments, all the architectures were designed in pyTorch. This was mainly for exposure using a different library.  
All NNs used ReLU activation functions where applicable.  

FFNN initialization  
![image](https://github.com/gitbheeds/EE399-Work/assets/87102849/36083cf3-71ab-43d7-876e-4303d3ee16d7)  

LSTM initialization  
![image](https://github.com/gitbheeds/EE399-Work/assets/87102849/aa8cca46-cae8-45ed-a2d6-d3682708fbd9)  

ESN initialization  
![image](https://github.com/gitbheeds/EE399-Work/assets/87102849/d7037959-cf63-4682-83e4-9039ec3046ad)  

RNN initialization  
![image](https://github.com/gitbheeds/EE399-Work/assets/87102849/45648c81-90d8-446b-9d99-b54de645ef96)
  

The Lorenz Equations were implemented using `generate_lorenz_data(rho)`, where `rho` indicates the value of &rho; for that specific set.  

![image](https://github.com/gitbheeds/EE399-Work/assets/87102849/96520193-46ae-4ed0-8eee-85b22b90b3fb)  

Since multiple values of &rho; were required for training and test data, a function was designed to generate all training data, train the model, generate the testing data, and calculate losses in both cases. The function, `train_and_test_model(rho_train_values, rho_test_values)` is shown below.  

![image](https://github.com/gitbheeds/EE399-Work/assets/87102849/595c822f-b967-4900-86d8-186a078e035c)  

This function prints to the terminal regularly to track the progress of the NN. An example of this output is shown below.  

![image](https://github.com/gitbheeds/EE399-Work/assets/87102849/dd0c85a9-130f-4eaf-872c-19bb18eb04b9)  


### Section 4: Results

After training and testing all the neural nets, we can determine the following order for performance. These numbers were taken from one run through all models, please note that loss results may differ each run of the NN due to the base randomness associated with a neural net. However, the overall performance rankings should be similar. 

1. ESN
  - &rho; = 17, loss of 24.25
  - &rho; = 35, loss of 73.16

2. LSTM
  - &rho; = 17, loss of 63.17
  - &rho; = 35, loss of 128.31

3. RNN
  - &rho; = 17, loss of 65.30
  - &rho; = 35, loss of 131.10

4. FFNN
  - &rho; = 17, loss of 113.07
  - &rho; = 35, loss of 428.52

### Section 5: Conclusion

In conclusion, we find that the ESN is best at forecasting the dynamics of the Lorenz equations. It's important to note though, that the losses associated with each &rho; value are still high, indicating the difficult in predicting the chaos of the Lorenz equations. We will start our analysis of the results by first considering the effects of &rho; on the system. To put it simply, &rho; affects the chaos of the system. Under the critical value of &rho; = 28, the system is relatively periodic, and thus can be predicted more accurately. Above the critical value, the system becomes incredibly chaotic and much harder to predict. Looking at our results, this checks out. Each neural net performed significantly worse when &rho; = 35. As this is above the critical threshold, it makes sense that the neural nets would struggle to predict this system effectively. Looking at the less chaotic test set, &rho; = 17, we see loss that is relatively lower than that of the other test set, but it is still relatively high compared to losses we have measured in other problems in past assignments. This is most likely because the neural nets have been trained on both stable and incredibly chaotic systems. This would likely decrease the model's ability to correctly predict low chaos systems.  

Now we turn our attention to the types of neural nets used, and why this ranking is feasible for them. First, we note that this system is time-dependent, and thus the data generated for them is time series data. Right off the bat, we can agree that the FFNN would perform the worst, as it is the only neural net we use that does not have a memory system, and therefore will struggle with time series data. The LSTM and ESN are both more advanced versions of the simpler RNN that were designed to combat some of the RNN's limitations. Standard RNNs suffer from the vanishing gradient problem, which makes it difficult for them to capture and propagate information over long sequences. LSTMs were specifically designed to alleviate this issue by incorporating memory cells and gating mechanisms. The LSTM's architecture allows it to selectively retain and update information over time, enabling it to capture long-term dependencies more effectively than a standard RNN. LSTMs and ESNs have more memory than RNNs by design, allowing them to handle complex long term dependencies better than their predecessor. When comparing why the ESN would outperform the LSTM, we first consider that the ESN has a simpler architecture and training procedure than the LSTM. Less computational demand allows ESNs to more efficiently process long datasets. ESNs require fewer parameters, decrease the amount of tuning required. ESNs have been shown to exhibit good robustness to noise and disturbances in the input data. The random initialization of the reservoir layer helps create a diverse set of echo states, which can help in capturing and representing noisy or corrupted input patterns. LSTMs, while generally robust, may require more careful tuning and regularization techniques to handle noisy inputs effectively. This robustness to noise may also help it handle chaotic input data more accurately. 


[Back to Table of Contents](https://github.com/gitbheeds/EE399-Work#table-of-contents) 


## HW6 Writeup: Analysis of Sea Temperature Data Using PyShred

### Abstract
This assignment takes the pySHRED repository published by Jan P. Williams, Olivia Zahn, and J. Nathan Kutz, and uses the Shallow Recurrent Decoder (SHRED) to map trajectories of sensor measurements into higher dimensions. We will assess the performace of the model as a function of two of its internal parameters, and then as a function of noise. 
### Section 1: Overview
In this assignment, first we will vary the time lag of the data from 52 to 100 and compute the performance as a function of time lag. The number of sensors will be kept at 3, as is provided in the example code. Then, we will keep the lag constant at 52 (the original value provided in the example code) and vary the number of sensors from 1 to 25, and plot the performance as a function of the number of sensors. Finally we will add Gaussian noise to the data at varying scale, and measure the SHRED's resistance to noise by measuring its performance. 

### Section 2: Background and Important Concepts

#### [pySHRED] (https://github.com/Jan-Williams/pyshred)
This repository contains the code for the paper "Sensing with shallow recurrent decoder networks" by Jan P. Williams, Olivia Zahn, and J. Nathan Kutz. SHallow REcurrent Decoders (SHRED) are models that learn a mapping from trajectories of sensor measurements to a high-dimensional, spatio-temporal state. For this task we used the preprocessing, training, testing and evaluation schemes from `example.ipynb`. The datasets considered in the paper "Sensing with shallow recurrent decoder networks" consist of sea-surface temperature (SST), a forced turbulent flow, and atmospheric ozone concentration. Cloning this repo will download the SST data used. Details for accessing the other datasets can be found in the supplement to the paper. For this assignment we will be looking at the SST data. 

##### LSTM 
LSTMs are a type of recurrent neural network (RNN) that can process sequential data by preserving information over time.

In an LSTM, the information can flow in multiple directions, as the output of the current time step can affect the input of the next time step. This means that LSTM layers generate feedback which can further be used to strengthen the LSTM. LSTMs have a unique architecture that allows them to capture long-term dependencies in the input sequence. They use a special memory cell that can maintain information over time and gates that control the flow of information into and out of the memory cell. The gates are composed of sigmoid activation functions that can decide which information to keep, forget or update in the memory cell. The architecture of an LSTM allows it to handle vanishing and exploding gradients that can occur in RNNs when training on long sequences. The gates can control the amount of gradient that flows through the network, which can prevent the gradient from vanishing or exploding. This specialized memory architecture allows LSTMs to better preserve long-term dependencies in the input data.  

##### Shallow Recurrent Decoder (SHRED) 
A shallow recurrent decoder refers to a type of decoder architecture that consists of a single layer of recurrent units. Recurrent units, such as Long Short-Term Memory (LSTM) or Gated Recurrent Unit (GRU), are specialized types of neural network layers that can capture and process sequential information effectively. In a shallow recurrent decoder, the recurrent units receive input from the encoded representation (latent space) and generate the output sequence or reconstruction. The output at each time step is fed back into the recurrent units as input for the next time step, allowing the decoder to capture dependencies and context within the sequence. Compared to deeper recurrent decoder architectures, shallow recurrent decoders have fewer layers and, consequently, fewer parameters. They are often used when the task at hand does not require a complex modeling of long-term dependencies. Shallow decoders are computationally lighter and faster to train, making them suitable for simpler sequence-to-sequence tasks. However, shallow recurrent decoders may struggle to capture complex patterns and dependencies present in the input sequence. Deeper architectures with multiple layers of recurrent units can potentially learn more intricate representations and better model long-range dependencies.

#### Gaussian Noise
Gaussian noise, also known as Gaussian distribution or white noise, is a type of random variation that follows a Gaussian or normal distribution. It is characterized by its probability density function, which forms a symmetric bell-shaped curve. Gaussian noise is often used in various fields of science and engineering to model random disturbances or errors in data. It is commonly added to signals or datasets to simulate real-world variability or to regularize models during training by adding stochasticity to the learning process.

### Section 3: Development and Implementation of Algorithms

#### PyShred
The pySHRED repo can be found [here](https://github.com/Jan-Williams/pyshred). We use the methods found in `example.ipynb` to load and process data, as well as train the models. The methods in `example.ipynb` are placed in a for loop and iterated through to generate performance data for different input parameters. An example is shown below.  
![image](https://github.com/gitbheeds/EE399-Work/assets/87102849/e486bd46-86d9-4b16-80e9-d87d34bb6a24)

#### Noise Generation
Noise was generated using the `np.random.normal()` function from numpy. Noise was added using a Gaussian distribution, and scaled by `level`, which iterated between 0.0 and 1.0 in 25 equal steps. The image below shows the implementation.  

![image](https://github.com/gitbheeds/EE399-Work/assets/87102849/addc194f-d86c-4aa4-949f-0716d336d13c)

### Section 4: Results

#### Performance vs Time Lag
![image](https://github.com/gitbheeds/EE399-Work/assets/87102849/45aa721e-2681-415e-b39a-a37804ba85d9)
#### Performance vs Number of Sensors
![image](https://github.com/gitbheeds/EE399-Work/assets/87102849/31611950-d064-4ed7-b13c-eb98678c67d7)
#### Performance vs Noise
![image](https://github.com/gitbheeds/EE399-Work/assets/87102849/3b59c408-c759-492a-8300-eb4d13716d94)


### Conclusion

As we analyze our results, a few things can be noticed. First and foremost, the SHRED has an incredibly high immunity to noise. The maximum loss when varying the noise was only 0.07, disregarding a large spike at 0.5 level. We also note that the more sensors the system has, the better it performs overall. This makes sense, as having more data from various locations in the sea can only improve the robustness of predictions. The performance compared to time lag was scattered at best, and didn't particularly follow any trends. Overall though, the performance was very good, reporting only about 0.0355 for the mean square error on average. 
