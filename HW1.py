from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# load the MNIST dataset
mnist = fetch_openml('mnist_784', version=1)

# apply PCA to reduce the dimensionality of the dataset
pca = PCA(n_components=4)
mnist_pca = pca.fit_transform(mnist.data)

# visualize the PCA-reduced dataset
plt.scatter(mnist_pca[:, 0], mnist_pca[:, 1], c=mnist.target.astype(int), cmap='jet')
plt.colorbar()
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()
