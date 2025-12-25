import numpy as np
import pandas as pd

# ------------PCA Implementation------------
# PCA using Eigenvalue Decomposition
class PCA_Scratch:
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.explained_variance = None
        self.explained_variance_ratio = None
        
    def fit(self, X):
        # 1. Mean centering
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        
        # 2. Covariance Matrix
        n_samples = X.shape[0]
        cov_matrix = (1 / (n_samples - 1)) * np.dot(X_centered.T, X_centered)
        
        # 3. Eigenvalue Decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # 4. Sort eigenvalues and eigenvectors in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Store full stats
        self.explained_variance = eigenvalues
        total_var = np.sum(eigenvalues)
        self.explained_variance_ratio = eigenvalues / total_var
        
        # 5. Select components
        if self.n_components:
            self.components = eigenvectors[:, :self.n_components]
        else:
            self.components = eigenvectors
            
        return self
    
    def transform(self, X):
        X_centered = X - self.mean
        return np.dot(X_centered, self.components)
    
    def inverse_transform(self, X_transformed):
        # Reconstruction: X_orig = X_trans * W.T + mean
        return np.dot(X_transformed, self.components.T) + self.mean

    def get_reconstruction_error(self, X):
        X_transformed = self.transform(X)
        X_reconstructed = self.inverse_transform(X_transformed)
        return np.mean((X - X_reconstructed) ** 2)

# ------------Autoencoder Implementation------------
# It includes a modular Layer system to handle the 3 hidden layers, manual backprop, and L2 regularization.

class Layer:
    def __init__(self, input_dim, output_dim, activation='relu'):
        # He Init for ReLU, Xavier for others
        if activation == 'relu':
            self.W = np.random.randn(input_dim, output_dim) * np.sqrt(2. / input_dim)
        else:
            self.W = np.random.randn(input_dim, output_dim) * np.sqrt(1. / input_dim)
        self.b = np.zeros((1, output_dim))
        self.activation = activation
        self.input = None
        self.z = None
        self.output = None
        self.dW = None
        self.db = None

    def forward(self, input_data):
        self.input = input_data
        self.z = np.dot(input_data, self.W) + self.b
        if self.activation == 'relu':
            self.output = np.maximum(0, self.z)
        elif self.activation == 'sigmoid':
            self.output = 1 / (1 + np.exp(-np.clip(self.z, -500, 500)))
        elif self.activation == 'tanh':
            self.output = np.tanh(self.z)
        elif self.activation == 'linear':
            self.output = self.z
        return self.output

    def backward(self, output_error, learning_rate, l2_reg):
        if self.activation == 'relu':
            activation_deriv = (self.z > 0).astype(float)
        elif self.activation == 'sigmoid':
            activation_deriv = self.output * (1 - self.output)
        elif self.activation == 'tanh':
            activation_deriv = 1 - np.power(self.output, 2)
        elif self.activation == 'linear':
            activation_deriv = 1.0
            
        d_z = output_error * activation_deriv
        
        # Gradients
        self.dW = np.dot(self.input.T, d_z)
        # L2 Regularization term added to gradient
        self.dW += l2_reg * self.W 
        self.db = np.sum(d_z, axis=0, keepdims=True)
        
        input_error = np.dot(d_z, self.W.T)
        
        # Update
        self.W -= learning_rate * self.dW
        self.b -= learning_rate * self.db
        
        return input_error

class Autoencoder_Scratch:
    def __init__(self, input_dim, bottleneck_dim, hidden_layers=[64, 32, 16], activation='relu'):
        self.layers = []
        # Encoder: Input -> H1 -> H2 -> ... -> Bottleneck
        dims = [input_dim] + hidden_layers
        for i in range(len(dims)-1):
            self.layers.append(Layer(dims[i], dims[i+1], activation))
        self.layers.append(Layer(dims[-1], bottleneck_dim, activation))
        
        # Decoder: Bottleneck -> ... -> H2 -> H1 -> Input
        dims = [bottleneck_dim] + hidden_layers[::-1]
        for i in range(len(dims)-1):
            self.layers.append(Layer(dims[i], dims[i+1], activation))
        # Final reconstruction layer (linear for standardization consistency)
        self.layers.append(Layer(dims[-1], input_dim, 'linear'))

    def forward(self, X):
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out
    
    def get_bottleneck(self, X):
        out = X
        # Encoder is half the layers
        encoder_len = len(self.layers) // 2
        for i in range(encoder_len):
            out = self.layers[i].forward(out)
        return out

    def fit(self, X, epochs=100, batch_size=32, learning_rate=0.01, l2_reg=0.0001):
        loss_history = []
        n_samples = X.shape[0]
        
        for epoch in range(epochs):
            # Shuffle
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            epoch_loss = 0
            
            # LR Scheduling: Decay every 20 epochs
            if epoch > 0 and epoch % 20 == 0:
                learning_rate *= 0.5
            
            for i in range(0, n_samples, batch_size):
                batch = X_shuffled[i:i+batch_size]
                
                # Forward
                output = self.forward(batch)
                
                # Loss (MSE)
                loss = np.mean((output - batch)**2)
                epoch_loss += loss
                
                # Backward (MSE derivative: 2*(Out - Target)/N)
                output_error = 2 * (output - batch) / batch.shape[0]
                for layer in reversed(self.layers):
                    output_error = layer.backward(output_error, learning_rate, l2_reg)
            
            avg_loss = epoch_loss / (n_samples / batch_size)
            loss_history.append(avg_loss)
            
        return loss_history
    
# ------------K-means Implementation------------
class KMeans_Scratch:
    def __init__(self, n_clusters=2, max_iter=300, tol=1e-4, init='k-means++'):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.init = init
        self.centroids = None
        self.labels = None
        self.inertia_ = None
        self.inertia_history = []

    def _initialize_centroids(self, X):
        n_samples, n_features = X.shape
        
        if self.init == 'random':
            indices = np.random.choice(n_samples, self.n_clusters, replace=False)
            centroids = X[indices]
        elif self.init == 'k-means++':
            # 1. Choose first center randomly
            centroids = [X[np.random.randint(n_samples)]]
            
            for _ in range(1, self.n_clusters):
                # 2. Calculate dist^2 to nearest existing center for all points
                dists_sq = np.array([min([np.inner(c-x, c-x) for c in centroids]) for x in X])
                
                # 3. Choose new center with probability proportional to dist^2
                probs = dists_sq / dists_sq.sum()
                cumprobs = probs.cumsum()
                r = np.random.rand()
                
                for i, p in enumerate(cumprobs):
                    if r < p:
                        centroids.append(X[i])
                        break
            centroids = np.array(centroids)
        return centroids

    def fit(self, X):
        self.centroids = self._initialize_centroids(X)
        self.inertia_history = []
        
        for i in range(self.max_iter):
            old_centroids = self.centroids.copy()
            
            # Expectation: Assign labels
            # Compute euclidean distance (broadcasting)
            # dist shape: (n_samples, n_clusters)
            distances = np.sqrt(((X[:, np.newaxis] - self.centroids) ** 2).sum(axis=2))
            self.labels = np.argmin(distances, axis=1)
            
            # Calculate Inertia (Sum of squared distances to closest centroid)
            min_dists = np.min(distances, axis=1)
            self.inertia_ = np.sum(min_dists ** 2)
            self.inertia_history.append(self.inertia_)
            
            # Maximization: Update centroids
            for k in range(self.n_clusters):
                if np.sum(self.labels == k) > 0:
                    self.centroids[k] = X[self.labels == k].mean(axis=0)
            
            # Convergence check
            shift = np.sum((self.centroids - old_centroids) ** 2)
            if shift < self.tol:
                break
                
        return self

    def predict(self, X):
        distances = np.sqrt(((X[:, np.newaxis] - self.centroids) ** 2).sum(axis=2))
        return np.argmin(distances, axis=1)

# ------------GMM Implementation------------
#EM algorithm with numerical stability checks and covariance support.
class GMM_Scratch:
    def __init__(self, n_components=2, max_iter=100, tol=1e-4, cov_type='full'):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.cov_type = cov_type
        self.means = None
        self.covs = None
        self.weights = None
        self.log_likelihood_history = []

    def _init_params(self, X):
        n_samples, n_features = X.shape
        # Initialize means randomly
        self.means = X[np.random.choice(n_samples, self.n_components, replace=False)]
        self.weights = np.full(self.n_components, 1 / self.n_components)
        
        # Initialize covariances
        if self.cov_type == 'full':
            self.covs = np.array([np.eye(n_features) for _ in range(self.n_components)])
        elif self.cov_type == 'tied':
            self.covs = np.eye(n_features)
        elif self.cov_type == 'diag':
            self.covs = np.ones((self.n_components, n_features))
        elif self.cov_type == 'spherical':
            self.covs = np.ones(self.n_components)

    def _estimate_log_prob(self, X):
        n_samples, n_features = X.shape
        log_probs = np.zeros((n_samples, self.n_components))
        const = -0.5 * n_features * np.log(2 * np.pi)

        for k in range(self.n_components):
            diff = X - self.means[k]
            
            if self.cov_type == 'full':
                # Regularize
                cov = self.covs[k] + np.eye(n_features) * 1e-6
                try:
                    L = np.linalg.cholesky(cov)
                    log_det = 2 * np.sum(np.log(np.diag(L)))
                    # Mahal: (x-u)T Sigma^-1 (x-u) solved via Cholesky
                    y = np.linalg.solve(L, diff.T)
                    mahal = np.sum(y**2, axis=0)
                except np.linalg.LinAlgError:
                    # Fallback for singular matrix
                    mahal = np.sum(diff**2, axis=1) # Treat as spherical identity
                    log_det = 0
                
            elif self.cov_type == 'tied':
                cov = self.covs + np.eye(n_features) * 1e-6
                L = np.linalg.cholesky(cov)
                log_det = 2 * np.sum(np.log(np.diag(L)))
                y = np.linalg.solve(L, diff.T)
                mahal = np.sum(y**2, axis=0)
                
            elif self.cov_type == 'diag':
                cov = self.covs[k] + 1e-6
                log_det = np.sum(np.log(cov))
                mahal = np.sum((diff**2) / cov, axis=1)
                
            elif self.cov_type == 'spherical':
                cov = self.covs[k] + 1e-6
                log_det = n_features * np.log(cov)
                mahal = np.sum(diff**2, axis=1) / cov
                
            log_probs[:, k] = const - 0.5 * (log_det + mahal)
            
        return log_probs + np.log(self.weights + 1e-10)

    def _e_step(self, X):
        weighted_log_prob = self._estimate_log_prob(X)
        log_prob_norm = np.logaddexp.reduce(weighted_log_prob, axis=1)
        # Subtract max for stability before exp
        log_resp = weighted_log_prob - log_prob_norm[:, np.newaxis]
        return np.exp(log_resp), log_prob_norm

    def _m_step(self, X, resp):
        n_samples, n_features = X.shape
        nk = resp.sum(axis=0) + 1e-10
        
        self.weights = nk / n_samples
        self.means = np.dot(resp.T, X) / nk[:, np.newaxis]
        
        if self.cov_type == 'full':
            for k in range(self.n_components):
                diff = X - self.means[k]
                # Weighted covariance
                self.covs[k] = np.dot(resp[:, k] * diff.T, diff) / nk[k]
                
        elif self.cov_type == 'tied':
            self.covs = np.zeros((n_features, n_features))
            for k in range(self.n_components):
                diff = X - self.means[k]
                self.covs += np.dot(resp[:, k] * diff.T, diff)
            self.covs /= n_samples
            
        elif self.cov_type == 'diag':
            avg_X2 = np.dot(resp.T, X * X) / nk[:, np.newaxis]
            avg_means2 = self.means ** 2
            self.covs = avg_X2 - avg_means2
            
        elif self.cov_type == 'spherical':
            avg_X2 = np.dot(resp.T, np.sum(X * X, axis=1)) / nk
            self.covs = (avg_X2 - np.sum(self.means**2, axis=1)) / n_features

    def fit(self, X):
        self._init_params(X)
        self.log_likelihood_history = []
        prev_log_lik = -np.inf
        
        for i in range(self.max_iter):
            resp, log_prob_norm = self._e_step(X)
            log_lik = np.sum(log_prob_norm)
            self.log_likelihood_history.append(log_lik)
            
            if abs(log_lik - prev_log_lik) < self.tol:
                break
            prev_log_lik = log_lik
            self._m_step(X, resp)
        return self

    def predict(self, X):
        resp, _ = self._e_step(X)
        return np.argmax(resp, axis=1)

    def get_bic_aic(self, X):
        n_samples, n_features = X.shape
        ll = self.log_likelihood_history[-1]
        
        # Count parameters
        if self.cov_type == 'full':
            n_params = self.n_components * (n_features * (n_features + 1) / 2)
        elif self.cov_type == 'diag':
            n_params = self.n_components * n_features
        elif self.cov_type == 'tied':
            n_params = n_features * (n_features + 1) / 2
        else: # spherical
            n_params = self.n_components
            
        n_params += self.n_components * n_features + (self.n_components - 1)
        
        bic = -2 * ll + n_params * np.log(n_samples)
        aic = -2 * ll + 2 * n_params
        return bic, aic