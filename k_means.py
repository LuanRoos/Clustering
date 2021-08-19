import numpy as np

class k_means:
	'''
	Class implementing k-means clustering using lloyd's algorithm.
	'''
	def __init__(self, n_clusters=8, init=None, max_iter=300):
		'''
		Parameters:
		n_clusters - Number of clusters to group.
		init - Optional parameter. A n_clustersxp ndarray containing initial cluster means. If not specified random labels will initially be assigned to each observation.
		max_iter - Default value of 300. Specifies how many iterations of lloyd's algorithm to run if convergence has not been reached.
		'''
		self.n_clusters = n_clusters
		self.max_iter = max_iter
		if init is not None:
			if init.shape[0] != n_clusters:
				print('Number of initial means provided does not match n_clusters')
				exit()
		self.cluster_means = init

	def fit(self, X):
		'''
		Fits the model on the nxp ndarray X.
		'''
		self.X = X.copy()
		if self.cluster_means is None:
			self.cluster_means = np.zeros((self.n_clusters, X.shape[1]), dtype=float)
			self.labels = np.random.randint(0, self.n_clusters, X.shape[0])
			for i in range(self.n_clusters):
				X_ = X[self.labels==i]
				if X_.shape[0] == 0:
					continue
				self.cluster_means[i] = np.mean(X_, axis=0, keepdims=True)
		else:
			self.labels = np.repeat(-1, X.shape[0])
		self.cluster_means = self.cluster_means.astype(float)
		for i in range(self.max_iter):
			before_labels = self.labels.copy()
			self.update_labels(X)
			if np.array_equal(before_labels, self.labels):
				return
			self.update_means(X)

	def update_labels(self, X):
		for i in range(X.shape[0]):
			distances = np.linalg.norm(self.cluster_means - X[i, np.newaxis], axis=1)
			self.labels[i] = np.argmin(distances)
		
	def update_means(self, X):
		for i in range(self.n_clusters):
			X_ = X[self.labels==i]
			if X_.shape[0] == 0:
				continue
			self.cluster_means[i] = np.mean(X_, axis=0, keepdims=True)

	def predict(self, X):
		'''
		Assigns the observations in the columns of the ndarray X to the most probable classes using the fitted model.
		'''
		pred_labels = np.empty(X.shape[0], dtype=int)
		for i in range(X.shape[0]):
			distances = np.linalg.norm(self.cluster_means - X[i, np.newaxis], axis=1)
			pred_labels[i] = np.argmin(distances)
		return pred_labels
	
	def get_cluster_covariances(self):
		covs = np.zeros((self.n_clusters, self.cluster_means.shape[1], self.cluster_means.shape[1]))
		for i in range(self.n_clusters):
			self.X[self.labels==i]
			X_ = self.X[self.labels==i] - self.cluster_means[i]
			if X_.shape[0] == 0:
				continue
			covs[i] = X_.T @ X_ / X_.shape[0]
		return covs
	
	def get_cluster_proportions(self):
		prop = np.empty(self.n_clusters)
		for i in range(self.n_clusters):
			prop[i] = np.count_nonzero(self.labels==i)
		return prop / np.sum(prop)
