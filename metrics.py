import numpy as np

# ------------Internal Metrics------------

# Computes Gap Statistic for k-means
def calculate_gap_statistic(X, k_range, model_class, n_references=5):
    gaps = []
    errors = []
    for k in k_range:
        # Real data
        model = model_class(n_clusters=k)
        model.fit(X)
        log_w = np.log(model.inertia_) # WCSS
        
        # Reference data
        ref_log_ws = []
        for _ in range(n_references):
            random_data = np.random.uniform(np.min(X, axis=0), np.max(X, axis=0), X.shape)
            model_ref = model_class(n_clusters=k)
            model_ref.fit(random_data)
            ref_log_ws.append(np.log(model_ref.inertia_))
            
        gap = np.mean(ref_log_ws) - log_w
        gaps.append(gap)
        s_k = np.std(ref_log_ws) * np.sqrt(1 + 1/n_references)
        errors.append(s_k)
    return gaps, errors

# How similar an object is to its own cluster compared to other clusters
def silhouette_score_scratch(X, labels):
    n_samples = X.shape[0]
    unique_labels = np.unique(labels)
    
    if len(unique_labels) < 2:
        return 0
        
    s_scores = np.zeros(n_samples)
    
    # Precompute distances
    dists = np.sqrt(np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2))
    
    for i in range(n_samples):
        # a(i): Mean distance to same cluster
        own_cluster = labels == labels[i]
        if np.sum(own_cluster) > 1:
            a_i = np.sum(dists[i, own_cluster]) / (np.sum(own_cluster) - 1)
        else:
            a_i = 0
            
        # b(i): Mean distance to nearest other cluster
        b_i = np.inf
        for label in unique_labels:
            if label == labels[i]:
                continue
            other_cluster = labels == label
            dist_to_other = np.mean(dists[i, other_cluster])
            b_i = min(b_i, dist_to_other)
            
        s_scores[i] = (b_i - a_i) / max(a_i, b_i)
        
    return np.mean(s_scores)

# Average similarity between each cluster and its most similar one
def davies_bouldin_index(X, labels):
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    
    centroids = np.array([X[labels == k].mean(axis=0) for k in unique_labels])
    
    # Average distance within cluster (scatter)
    S = np.zeros(n_clusters)
    for i, k in enumerate(unique_labels):
        diff = X[labels == k] - centroids[i]
        dist = np.sqrt(np.sum(diff**2, axis=1))
        S[i] = np.mean(dist)
        
    R = np.zeros(n_clusters)
    for i in range(n_clusters):
        max_val = -np.inf
        for j in range(n_clusters):
            if i != j:
                # Distance between centroids
                dist_centroid = np.linalg.norm(centroids[i] - centroids[j])
                val = (S[i] + S[j]) / dist_centroid
                if val > max_val:
                    max_val = val
        R[i] = max_val
        
    return np.mean(R)

def calinski_harabasz_scratch(X, labels):
    """
    Calculates Variance Ratio Criterion.
    CH = [trace(B) / (k - 1)] / [trace(W) / (n - k)]
    """
    n_samples, n_features = X.shape
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    
    if n_clusters < 2:
        return 0.0
        
    extra_dispersion = 0.0
    intra_dispersion = 0.0
    mean = np.mean(X, axis=0)
    
    for k in unique_labels:
        cluster_k = X[labels == k]
        mean_k = np.mean(cluster_k, axis=0)
        # Extra-cluster dispersion (Between-group sum of squares)
        extra_dispersion += len(cluster_k) * np.sum((mean_k - mean) ** 2)
        # Intra-cluster dispersion (Within-group sum of squares)
        intra_dispersion += np.sum((cluster_k - mean_k) ** 2)
        
    return (extra_dispersion * (n_samples - n_clusters)) / (intra_dispersion * (n_clusters - 1))

# ------------External Metrics------------
# Measures the extent to which clusters contain a single class
def purity_score(y_true, y_pred):
    # Contingency matrix
    contingency = np.zeros((len(np.unique(y_true)), len(np.unique(y_pred))))
    # We must map labels to 0..N indices for matrix
    u_true = np.unique(y_true)
    u_pred = np.unique(y_pred)
    map_true = {val: i for i, val in enumerate(u_true)}
    map_pred = {val: i for i, val in enumerate(u_pred)}
    
    for t, p in zip(y_true, y_pred):
        contingency[map_true[t], map_pred[p]] += 1
        
    return np.sum(np.amax(contingency, axis=0)) / np.sum(contingency)

def adjusted_rand_index_scratch(labels_true, labels_pred):
    # Implementing contingency logic manually:
    classes = np.unique(labels_true)
    clusters = np.unique(labels_pred)
    
    tp_plus_fp = 0 # sum of (n_cluster choose 2)
    tp_plus_fn = 0 # sum of (n_class choose 2)
    tp = 0         # sum of (n_ij choose 2)
    
    # Contingency table
    contingency = np.zeros((len(classes), len(clusters)))
    for i, t in enumerate(classes):
        for j, p in enumerate(clusters):
            contingency[i, j] = np.sum((labels_true == t) & (labels_pred == p))
            
    # Comb(n, 2) = n*(n-1)/2
    def comb2(n): return n * (n - 1) / 2
    
    sum_ij = 0
    for i in range(len(classes)):
        for j in range(len(clusters)):
            sum_ij += comb2(contingency[i, j])
            
    sum_a = sum([comb2(np.sum(contingency[i, :])) for i in range(len(classes))])
    sum_b = sum([comb2(np.sum(contingency[:, j])) for j in range(len(clusters))])
    
    n_samples = len(labels_true)
    total_comb = comb2(n_samples)
    
    expected_index = (sum_a * sum_b) / total_comb
    max_index = (sum_a + sum_b) / 2
    
    if max_index == expected_index:
        return 1
        
    return (sum_ij - expected_index) / (max_index - expected_index)

def entropy(labels):
    """Calculates entropy of a labeling."""
    n = len(labels)
    if n == 0: return 0.0
    probs = np.array([np.sum(labels == k) / n for k in np.unique(labels)])
    return -np.sum(probs * np.log(probs + 1e-10))

def mutual_info(labels_true, labels_pred):
    """Calculates mutual information between two labelings."""
    classes = np.unique(labels_true)
    clusters = np.unique(labels_pred)
    n = len(labels_true)
    mi = 0.0
    
    for c in classes:
        for k in clusters:
            # Joint probability p(c, k)
            intersect = np.sum((labels_true == c) & (labels_pred == k))
            if intersect == 0: continue
            
            p_ck = intersect / n
            p_c = np.sum(labels_true == c) / n
            p_k = np.sum(labels_pred == k) / n
            
            mi += p_ck * np.log(p_ck / (p_c * p_k) + 1e-10)
    return mi

def normalized_mutual_info_scratch(labels_true, labels_pred):
    """Calculates NMI: 2 * MI / (H(true) + H(pred))"""
    h_true = entropy(labels_true)
    h_pred = entropy(labels_pred)
    mi = mutual_info(labels_true, labels_pred)
    return 2 * mi / (h_true + h_pred + 1e-10)
