# EE399-Work
Storage for EE399 HW
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
- HW4 Writeup: Training Neural Nets
  - Abstract
  - Overview
  - Background and Important Concepts
  - Development and Implementation of Algorithms
  - Results
  - Conclusion 
    
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






