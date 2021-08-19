import numpy as np
from k_means import k_means

class GMM:
	'''
	A class to fit Gaussian Mixture Models using an Expectation Maximisation algorithm.
	'''
	def __init__(self, n_components=1, max_iter=300, means=None, covs=None, mix_weights=None):
		if means is not None:
			if means.shape[0] != n_components:
				print('Number of initial means provided does not match n_components.')
				exit()
		if covs is not None:
			if covs.shape[0] != n_components:
				print('Number of covariance matrices provided does not match n_components.')
				exit()
		if mix_weights is not None:
			if mix_weights.shape[0] != n_components:
				print('Number of mixing weights provided does not match n_components.')
				exit()

		self.n_components = n_components
		self.max_iter = max_iter
		self.means = means
		self.covs = covs
		self.mix_weights = mix_weights

	def fit(self, X):
		self.responsibilities = np.empty((X.shape[0], self.n_components))
		if self.means is None or self.covs is None or self.mix_weights is None:
			km = k_means(self.n_components, max_iter = 10)
			km.fit(X)
		if self.means is None:
			self.means = km.cluster_means
		if self.covs is None:
			self.covs = km.get_cluster_covariances()
		if self.mix_weights is None:
			self.mix_weights = km.get_cluster_proportions()
		self.means = self.means.astype(float)
		self.covs = self.covs.astype(float)
		self.mix_weights = self.mix_weights.astype(float)
		for i in range(self.max_iter):
			self.responsibilities = self.calc_responsibilities(X)
			self.update_params(X)
	
	def update_params(self, X):
		N_ = np.sum(self.responsibilities, axis=0, keepdims=True).T
		self.means = self.responsibilities.T @ X / N_
		for i in range(self.n_components):
			diag_respo = np.diag(self.responsibilities[:, i])
			V = (X - self.means[i])
			self.covs[i] = V.T @ diag_respo @ V / N_[i]
		self.mix_weights = N_/X.shape[0]
		
	def calc_responsibilities(self, X):
		responsibilities = np.empty((X.shape[0], self.n_components), dtype=float)
		for i in range(X.shape[0]):
			for j in range(self.n_components):
				responsibilities[i, j] = np.log(self.mix_weights[j]) + GMM.log_norm_density(X[i], self.means[j], self.covs[j])
			responsibilities[i] = GMM.softmax(responsibilities[i])
		return responsibilities 
	
	def softmax(x):
    		e_x = np.exp(x - np.max(x))
    		return e_x / e_x.sum()

	def predict_class(self, X):
		return np.argmax(self.calc_responsibilities(X), axis=1)

	'''
	Expects x and mu of dimension (p, ), cov of dimension (p, p)
	'''
	def norm_density(x, mu=None, cov=None):
		if mu is None:
			mu = np.zeros(x.shape[0])
		if cov is None:
			cov = np.identity(x.shape[0])
		return np.exp(GMM.log_norm_density(x, mu, cov))

	'''
	Expects x and mu of dimension (p, ), cov of dimension (p, p)
	'''
	def log_norm_density(x, mu=None, cov=None):
		if mu is None:
			mu = np.zeros(x.shape[0])
		if cov is None:
			cov = np.identity(x.shape[0])

		x = x[:, np.newaxis]
		mu = mu[:, np.newaxis]
		trans = x - mu
		mal = -trans.T.dot(np.linalg.solve(cov, trans))/2.0
		covdet = np.linalg.det(2.*np.pi*cov)
		logf = -0.5*np.log(covdet) + mal
		return logf
