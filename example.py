import numpy as np
import matplotlib.pyplot as plt
from k_means import k_means
from gmm import GMM

# Generate random data belonging to 4 classes. Plot the data and strip the class information from each observation.
# Then perform clustering, and hopefully the old class information will have been recovered.

N = 1000
K = 4
true_labels = np.random.randint(0, K, N)
true_means = np.random.rand(K, 2)*10
covs = np.empty((K, 2, 2))
for i in range(K):
	cov_ = np.random.rand(2)
	covs[i] = np.outer(cov_, cov_)
	
X = np.empty((N, 2))
for i in range(N):
	X[i] = np.random.multivariate_normal(true_means[true_labels[i]], np.identity(2))

plt.subplot(3, 1, 1)
plt.scatter(X[:, 0], X[:, 1], c=true_labels)
plt.scatter(true_means[:, 0], true_means[:, 1], c='Red')
plt.title('True labels')

print('Running...')

km = k_means(n_clusters=K, max_iter=100)
km.fit(X)
plt.subplot(3, 1, 2)
plt.scatter(X[:, 0], X[:, 1], c=km.predict(X))
plt.scatter(km.cluster_means[:, 0], km.cluster_means[:, 1], c='Red')
plt.title('K-means clustering labels')

gm = GMM(n_components=K, max_iter=100)
gm.fit(X)
plt.subplot(3, 1, 3)
plt.scatter(X[:, 0], X[:, 1], c=gm.predict_class(X))
plt.scatter(gm.means[:, 0], gm.means[:, 1], c='Red')
plt.title('GMM clustering labels')

print('Complete!')

plt.show()
