import numpy as np
import warnings

from scipy import  stats,sparse,spatial,linalg
from sklearn import  decomposition,  preprocessing,neighbors
import pandas as pd

warnings.simplefilter(action='ignore', category=UserWarning)



def gauss_kernel(data, k_neighbors, delta=2, alpha=1):
    """
    Creates a Gaussian kernel matrix using nearest neighbors, with normalization.
    
    Parameters:
        data: Input data (n_samples x n_features)
        k_neighbors: Number of nearest neighbors to consider
        delta: Scaling factor (default=2)
        alpha: Exponent for power law (default=1)
    
    Returns:
        weight_matrix: Normalized kernel matrix
        knn_indices: Indices of nearest neighbors for each point
        sqsigmas: Squared bandwidth for each point (sigma^2)
        sigs_mul: Matrix of sigma_i * sigma_j products
    """
    
    n_samples = data.shape[0]  # Number of data points

    # Find k-nearest neighbors for each point using squared Euclidean distance
    nbrs = neighbors.NearestNeighbors(n_neighbors=k_neighbors, metric='sqeuclidean').fit(data)
    knn_dists, knn_indices = nbrs.kneighbors(data)  

    sqsigmas = knn_dists[:, -1]  
    sigmas = np.sqrt(sqsigmas)   

    # Precompute all pairwise products of sigma's (sigma_i * sigma_j)
    # sigs_mul = np.multiply.outer(sigmas, sigmas)

    # Build Gaussian kernel weights for each neighbor
    kernel = np.zeros((n_samples, k_neighbors))
    for i in range(n_samples):
        # power_law = np.power(knn_dists[i, :] / (delta * (sigs_mul[i, knn_indices[i, :]])), alpha)
        power_law = np.power(knn_dists[i, :] / (delta * sigmas[i] * sigmas[knn_indices[i,:]]), alpha)
        
        kernel[i, :] = np.exp(-power_law)

    # Create sparse kernel matrix (make sure it's symmetric)
    indptr = range(0, (n_samples + 1) * k_neighbors, k_neighbors)
    k_matrix = sparse.csr_matrix((kernel.flatten(), knn_indices.flatten(), indptr), shape=(n_samples, n_samples))

    k_matrix_T = k_matrix.transpose()
    prod_matrix = k_matrix.multiply(k_matrix_T)
    kernel_matrix = k_matrix + k_matrix_T - prod_matrix   

    # Normalize the kernel matrix (D^(-1/2) * K * D^(-1/2))
    k_d = np.sqrt(np.asarray(kernel_matrix.sum(axis=0)))  
    kd_inv_sq = sparse.spdiags(1.0 / k_d, 0, n_samples, n_samples)  
    weight_matrix = kd_inv_sq @ kernel_matrix @ kd_inv_sq  

    return weight_matrix, knn_indices, sigmas

def box(d,local_sigma):
    """
    Computes the box kernel (also known as the rectangular kernel).

    This kernel assigns a value of 1 if the distance is within a given threshold (local_sigma),
    and 0 otherwise. It's commonly used in simple forms of density estimation.

    Args:
        d (numpy.ndarray or float): The input distances.
        local_sigma (float): The threshold value (cutoff distance).

    Returns:
        numpy.ndarray or float: 
        The box kernel values. Returns 1 if the distance is within local_sigma, 0 otherwise.

    """
    return np.where(d <= local_sigma, 1, 0)

def box_kernel(data, k_neighbors):
    """    
    Computes a box kernel matrix for the given data using the k-nearest neighbors.
    In this kernel, all distances to the k-nearest neighbors are assigned a value of 1, 
    meaning that each data point is equally weighted within its neighborhood. And then normalize the kernel matrix.

    Args:
        data (np.ndarray):  The input data matrix with shape (n_samples, n_features).
        k_neighbors (int):  The number of nearest neighbors to consider for each data point.

    Returns:
        kernel_tilde (scipy.sparse.csr_matrix): The normalized box kernel matrix, where connections between neighbors are weighted by degree normalization.
        knn_indices (np.ndarray):  The indices of the k nearest neighbors for each data point.
    """

    n_samples = data.shape[0] 
    nbrs = neighbors.NearestNeighbors(n_neighbors = k_neighbors, metric='euclidean',n_jobs = -2).fit(data)
    knn_dists,knn_indices = nbrs.kneighbors(data)

    sqsigmas = knn_dists[:, -1]  
    sigmas = np.sqrt(sqsigmas)   
    
    kernel = np.ones((n_samples,k_neighbors))
    
    indptr = range(0,(n_samples+1)*k_neighbors,k_neighbors)
    k_matrix = sparse.csr_matrix((kernel.flatten(),knn_indices.flatten(),indptr),shape=(n_samples,n_samples))
    kernel_matrix = k_matrix.maximum(k_matrix.T) 

    k_d = np.sqrt(np.asarray(kernel_matrix.sum(axis=0)))
    kd_inv_sq = sparse.spdiags(1.0 / k_d, 0, n_samples, n_samples)
    kernel_tilde = kd_inv_sq @ kernel_matrix @ kd_inv_sq
 
    # kd_t = np.sqrt(np.asarray(kernel_tilde.sum(axis=0)))
    # D_inv_sq = sparse.spdiags(1.0 / kd_t, 0, n_samples, n_samples)
    # norm_kernel = D_inv_sq @ kernel_tilde @ D_inv_sq

    return kernel_tilde,knn_indices,sigmas


def box_kernel2(data, k_neighbors, delta=1):
    """    
    Compute a kernel matrix using the box kernel and handle disconnected components using minimum spanning trees.
    This method builds a kernel matrix using a box function to define the influence of neighbors, and if disconnected 
    components exist, it uses a minimum spanning tree (MST) to connect them.

    Args:
        data: numpy.ndarray, shape (n_samples, n_features)
            The input data points.
        k_neighbors: int
            The number of nearest neighbors to consider for each data point.
        delta: float, optional (default=1)
            A scaling factor for local sigma computation that controls the bandwidth of the box kernel.

    Returns:
        tuple:
        - kernel_tilde (scipy.sparse.csr_matrix): 
            The normalized box kernel matrix.
        - knn_indices (np.ndarray): 
            The indices of the k nearest neighbors for each data point.
    """

    n_samples = data.shape[0] 

    # Compute the k nearest neighbors
    nbrs = neighbors.NearestNeighbors(n_neighbors=k_neighbors, metric='sqeuclidean',n_jobs = -2).fit(data)
    knn_dists, knn_indices = nbrs.kneighbors(data)

    # Compute the sigmas
    sigmas = np.sqrt(knn_dists[:, -1])

    # Compute the weight matrix using the box kernel
    kernel = np.zeros((n_samples, k_neighbors), dtype=np.float32)
    local_sigmas = np.zeros(n_samples)
    for i in range(n_samples):
        local_sigmas[i] = delta * sigmas[i] * sigmas[knn_indices[i, -1]]
        if local_sigmas[i] <= knn_dists[i,3]:
            local_sigmas[i] = knn_dists[i,3]
        kernel[i, :] = box(knn_dists[i, :], local_sigmas[i]) # box_kernel

    # Construct the sparse kernel matrix
    indptr = range(0, (n_samples + 1) * k_neighbors, k_neighbors)
    k_matrix = sparse.csr_matrix((kernel.flatten(), knn_indices.flatten(), indptr), shape=(n_samples, n_samples))
    kernel_matrix = k_matrix.maximum(k_matrix.T) 
    kernel_matrix.eliminate_zeros()

    n_components,labels = sparse.csgraph.connected_components(csgraph=kernel_matrix, directed=True, return_labels=True, connection= 'weak')
    if n_components > 1:
        indptr = range(0,(n_samples+1)*k_neighbors,k_neighbors)
        dist_matrix = sparse.csr_matrix((knn_dists.flatten(), knn_indices.flatten(), indptr), shape=(n_samples,n_samples))
        Tcsr = sparse.csgraph.minimum_spanning_tree(dist_matrix)
        Tcsr = Tcsr.maximum(Tcsr.T)
        kernel_matrix = kernel_matrix.maximum(Tcsr) 

    # Compute the diagonal normalization matrix
    k_d = np.sqrt(np.asarray(kernel_matrix.sum(axis=0)))
    kd_inv_sq = sparse.spdiags(1.0 / k_d, 0, n_samples, n_samples)
    kernel_tilde = kd_inv_sq @ kernel_matrix @ kd_inv_sq

    # Compute the row normalization matrix
    # kd_t = np.sqrt(np.asarray(kernel_tilde.sum(axis=0)))
    # D_inv_sq = sparse.spdiags(1.0 / kd_t, 0, n_samples, n_samples)
    # norm_kernel = D_inv_sq @ kernel_tilde @ D_inv_sq

    return kernel_tilde,knn_indices,sigmas

def mix_decay(d,local_sigma,alpha=1):
    """
    The function returns 1 for distances less than or equal to local_sigma, and applies
    an exponential Gaussian decay for larger distances. This can be useful when modeling
    a smooth decay of influence with distance.

    Args:
        d (numpy.ndarray or float): The input distances.
        local_sigma (float): The cutoff or threshold distance for switching between constant and decayed values.
        alpha (float, optional): The decay rate. Defaults to 1.

    Returns:
        numpy.ndarray or float: 
        The mixed decayed values, where distances less than local_sigma will return 1, and larger distances follow a decayed Gaussian function.
    """
    return np.where( d <= local_sigma, 1, np.exp(- np.power(d/local_sigma,alpha)))

def mix_kernel(data, k_neighbors, delta=1, alpha=1):
    """    
    Compute a mixed kernel matrix using a combination of box and Gaussian decay kernels.

    Args:
        data (array-like, shape (n_samples, n_features)): 
            The input data points.
        k_neighbors (int): 
            The number of nearest neighbors to consider for each point.
        delta (float, optional, default=1): 
            Scaling factor for computing the local sigma (spread parameter) for the decay.
        alpha (float, optional, default=1): 
            Parameter for the Gaussian decay function that controls the rate of decay.

    Returns:
        tuple:
        - kernel_tilde (scipy.sparse.csr_matrix): 
            The normalized, symmetric kernel matrix based on the nearest neighbors.
        - knn_indices (np.ndarray): 
            Indices of the k nearest neighbors for each point.

    """

    n_samples = data.shape[0] 

    # Compute nearest neighbors
    nbrs = neighbors.NearestNeighbors(n_neighbors=k_neighbors, metric='sqeuclidean').fit(data)
    knn_dists, knn_indices = nbrs.kneighbors(data)

    # Compute weight matrix
    sigmas = np.sqrt(knn_dists[:, -1])
    kernel = np.zeros((n_samples, k_neighbors))
    local_sigmas = np.zeros(n_samples)
    for i in range(n_samples):
        local_sigmas[i] = delta * sigmas[i] * sigmas[knn_indices[i, -1]]
        kernel[i, :] = mix_decay(knn_dists[i, :], local_sigmas[i], alpha) # gauss_decay

    # Create the kernel matrix
    indptr = range(0, (n_samples + 1) * k_neighbors, k_neighbors)
    k_matrix = sparse.csr_matrix((kernel.flatten(), knn_indices.flatten(), indptr), shape=(n_samples, n_samples))
    kernel_matrix = k_matrix.maximum(k_matrix.T) 

    # Normalize the kernel matrix
    k_d = np.sqrt(np.asarray(kernel_matrix.sum(axis=0)))
    kd_inv_sq = sparse.spdiags(1.0 / k_d, 0, n_samples, n_samples)
    kernel_tilde = kd_inv_sq @ kernel_matrix @ kd_inv_sq

    # kd_t = np.sqrt(np.asarray(kernel_tilde.sum(axis=0)))
    # D_inv_sq = sparse.spdiags(1.0 / kd_t, 0, n_samples, n_samples)
    # norm_kernel = D_inv_sq @ kernel_tilde @ D_inv_sq

    return kernel_tilde,knn_indices,sigmas


def scanpy_kernel(data, knn=5, method='umap'):
    """
    This function creates a kernel matrix using scanpy and graph-tool libraries.

    Args:
        data: A numpy array (n_samples, n_features) representing the data to be used for kernel construction.
        knn: The number of nearest neighbors to consider when constructing the adjacency matrix (default: 5).
        method: The dimensionality reduction method to use for neighbor search (default: 'umap').
                Other possible values could be 'gauss' or 'pca', depending on Scanpy's implementation.

    Returns:
        A kernel matrix represented as a sparse matrix from Graph-tools.
    """
 
    try:
        import scanpy as sc
        import graphtools
    except ImportError as imp_err:
        sc = imp_err
        graphtools = imp_err

    adata = sc.AnnData(data)
    sc.pp.neighbors(adata, n_neighbors=knn,method=method)
    Dists = adata.obsp["distances"]
    W = adata.obsp["connectivities"]
    K = graphtools.matrix.set_diagonal(W, 1)
    return K,None,Dists


def phate_kernel(data, knn = 5, decay = 40.0, anisotropy = 0, **kwargs):
    """
    This function creates a kernel matrix using the PHATE method with the help of the graph-tool library.

    Args:
        data: A numpy array (n_samples, n_features) representing the data to be used for kernel construction.
        knn: The number of nearest neighbors to consider when constructing the graph (default: 5).
        decay: The decay parameter that controls the influence of neighboring points (default: 40.0). 
               Higher decay values lead to smoother kernels by controlling the decay of the kernel weights.
        anisotropy: The anisotropy parameter that controls the influence of points in different directions (default: 0). 
                    Non-zero values introduce direction-based weighting into the kernel.
        n_pca: The number of principal components to use for dimensionality reduction before building the graph 
               (default: None, meaning it uses all components).
        **kwargs: Additional keyword arguments passed to the graph-tool.Graph constructor (optional).

    Returns:
        K: A kernel matrix represented as a sparse matrix from graph-tool.
    """
    try:
        import graphtools
    except ImportError as imp_err:
        graphtools = imp_err

    G = graphtools.Graph(data,knn=knn,decay=decay,anisotropy=anisotropy,use_pygsp=True,random_state=0)
    K = G.kernel
    sigmas = G.bandwidth
    return K,None,sigmas




def eigen_kernel(kernel):
    """
    Computes eigenvalues and eigenvectors of a normalized kernel matrix.
    
    Parameters:
        kernel: Input kernel matrix (symmetric and normalized)
    
    Returns:
        Phi: Left eigenvectors (scaled)
        lamb: Eigenvalues (sorted in descending order)
        Psi: Right eigenvectors (scaled)
    """
    
    # Compute normalization factors (sum of each column)
    kernel_sum = kernel.sum(axis=0)  
    kd = np.sqrt(kernel_sum) 
    
    # Create diagonal normalization matrix
    ks = np.diag(1/kd)  
    
    # Compute normalized kernel matrix Mp = D^(-1/2)*K*D^(-1/2)
    Mp = ks @ kernel @ ks
    
    # Compute eigenvalues and eigenvectors
    [lamb, u] = linalg.eigh(Mp)  
    
    # Sort eigenvalues in descending order
    idx = lamb.argsort()[::-1]  
    lamb = lamb[idx]  
    u = u[:, idx]  
    
    # Ensure positive eigenvalues by flipping signs if needed
    v = u.copy()
    v[:, lamb < 0] = -u[:, lamb < 0] 
    lamb = abs(lamb)  
    
    # Compute scaled eigenvectors
    Phi = ks @ u  
    Psi = v.T @ np.diag(kd) 
    
    return Phi, lamb, Psi


def eigen_kernel2(matrix):
    """
    This function computes the eigenvalues, eigenvectors, and pseudoinverse of a kernel matrix.

    Args:
        matrix: A 2D numpy array representing the kernel matrix.

    Returns:
        A tuple containing three elements:
        lamb: A 1D numpy array containing the eigenvalues of the kernel matrix.
        Phi: A 2D numpy array containing the eigenvectors of the kernel matrix (one eigenvector per column).
        Psi: A 2D numpy array containing the pseudoinverse of the eigenvector matrix (Phi).
    """

    lamb, Phi = np.linalg.eig(matrix)
    Psi = np.linalg.inv(Phi)

    return Phi,lamb,Psi


def compute_landmark_operator(K, labels, random_state = None):
    """
    This function computes the landmark operator based on a kernel matrix, number of landmarks, and sample labels.

    Args:
        K: A sparse matrix representing the kernel matrix.
        labels: A 1D numpy array containing integer labels for each sample.
        random_state: An integer (optional) to control the randomness for landmark selection (default: None).
    Returns:
        A tuple containing two elements:
        pmm: A 2D numpy array representing the landmark operator.
        pnm: A 2D numpy array representing the intermediate matrix used in the calculation.
    """

    landmarks = np.unique(labels)
    kmn = sparse.vstack([sparse.csr_matrix(K[labels == i, :].sum(axis=0)) for i in landmarks])

    pmn = preprocessing.normalize(kmn,norm="l1",axis=1)
    pnm = preprocessing.normalize(kmn.T,norm="l1",axis=1)
    pmm = pmn @ pnm
    pmm = pmm.toarray()
    pnm = pnm.toarray()

    return pmm,pnm


def calc_l(lamb):
    """
    Determines the optimal cutoff point 'l' by analyzing the entropy changes of lambda powers.
    
    The function examines how the entropy changes across increasing powers of lambda
    to identify a natural cutoff point in the sequence.

    Args:
        lamb: A float value (typically between 0 and 1) representing the decay rate

    Returns:
        An integer representing the recommended cutoff index 'l'
    """
    
   
    dse_list = []
    for i in range(30):
        dse = stats.entropy(np.power(lamb, i))
        dse_list.append(dse)
    
    da = np.gradient(dse_list) 
    dda = np.gradient(da)       
    
    if np.sum(np.diff(np.sign(dda))) == 0:
        l = 2
    else:
        l = np.where(np.diff(np.sign(dda)) != 0)[0][0] + 2
    
    return l


def compute_transform_matrix(X, X_center, metric='sqeuclidean'):
    """
    """

    distance_matrix = spatial.distance.cdist(X, X_center, metric=metric)

    min_distances = np.min(distance_matrix, axis=1, keepdims=True)
    normalized_matrix = (distance_matrix - min_distances) / min_distances
    similarity_matrix = np.exp(-normalized_matrix)
    transform_matrix = preprocessing.normalize(similarity_matrix, norm='l1', axis=1)

    return transform_matrix


def compute_transform_matrix(X, X_center, X_label, knn_indices, sqsigmas, k_neighbors, delta=2, alpha=1, metric='sqeuclidean'):
    """
    Computes a transformation matrix that maps data points to cluster centers using a kernel-based approach.
    
    Args:
        X: Input data matrix (n_samples x n_features)
        X_center: Cluster centers (n_centers x n_features)
        X_label: Cluster label for each data point
        knn_indices: Nearest neighbor indices for each cluster
        sqsigmas: Squared bandwidth parameters (σ²) for each cluster
        k_neighbors: Number of nearest neighbors to consider
        delta: Scaling factor (default=2)
        alpha: Exponent for kernel (default=1)
        metric: Distance metric (default='sqeuclidean')
        
    Returns:
        transform_matrix: Normalized mapping matrix (n_samples x n_centers)
    """
    
    # Get dimensions
    n_samples = X.shape[0]  
    n_centers = X_center.shape[0]  

    # Initialize matrices
    xk_dists = np.zeros((n_samples, k_neighbors))  
    xk_kernel = np.zeros((n_samples, k_neighbors))  

    # Get bandwidth (σ²) for each point's assigned cluster
    X_sqsigmas = sqsigmas[X_label]  

    # Compute distances and kernel values to nearest centers
    for i in range(n_samples):
        xk_dists[i,:] = spatial.distance.cdist(
            X[i].reshape(1, -1),  
            X_center[knn_indices[X_label[i]]],  
            metric=metric
        )
        
        # Compute Gaussian kernel: exp(-(distance/(δ*σ))^α)
        xk_kernel[i,:] = np.exp(-np.power(
            xk_dists[i, :] / (delta * np.sqrt(X_sqsigmas[i])), 
            alpha
        ))

    # Create full kernel matrix (n_samples x n_centers)
    xm_kernel = np.zeros((n_samples, n_centers))
    for i in range(n_samples):
        # Only keep kernel values for the k nearest centers
        xm_kernel[i][knn_indices[X_label[i]]] = xk_kernel[i,:]

    # Normalize to create probability distribution over centers
    transform_matrix = preprocessing.normalize(xm_kernel, norm='l1', axis=1)

    return transform_matrix
    


def compute_infty_R(Phi, lamb, Psi, cv, l):
    """
    Computes the influence matrix R and its derivative using spectral decomposition.
    
    Args:
        Phi: Left eigenvectors (n_samples x n_components)
        lamb: Eigenvalues (n_components,)
        Psi: Right eigenvectors (n_components x n_samples)
        cv: damping factors vector (n_samples,)
        l: Power parameter (integer)
        
    Returns:
        R: Normalized influence matrix (n_samples x n_samples)
        dif_R: Derivative of influence matrix (n_samples x n_samples)
    """
    
    n_samples = len(cv)  # Number of data points

    # Compute the numerator and denominator terms for Sigma
    lamb_l = np.power(lamb, l)  
    ncl = np.outer(1 - cv, lamb_l)  
    dcl = 1 - np.outer(cv, lamb)  

    # Compute the Sigma matrix
    Sigma = ncl / dcl 
    
    # Compute the influence matrix R = Φ * Σ * Ψ
    R = Phi * Sigma @ Psi  # * is element-wise multiplication, @ is matrix multiplication

    # Compute the derivative components
    dd_f = np.power(dcl, 2)  
    lam_f = lamb_l * (lamb - 1)  
    nd_f = np.tile(lam_f, (n_samples, 1))  
    dSigma = nd_f / dd_f  
    
    # Compute derivative matrix dif_R = Φ * dΣ * Ψ
    dif_R = Phi * dSigma @ Psi

    # Post-processing
    R[R < 0] = 0  
    dif_R[R == 0] = 0  
    
    # Normalize rows to sum to 1
    R = preprocessing.normalize(R, norm="l1", axis=1)
    
    return R, dif_R




def classic(D, n_components=2, random_state=None):
    """
    Performs classic multidimensional scaling (MDS) on a distance matrix D.
    
    This implementation centers the distance matrix and then applies PCA to 
    obtain the init low-dimensional embedding.

    Args:
        D: Input distance matrix (n_samples x n_samples)
        n_components: Number of dimensions for the output embedding (default=2)
        random_state: Random seed for reproducibility (default=None)

    Returns:
        Y: Embedded coordinates (n_samples x n_components)
    """
    
    # Double-center the distance matrix. Subtract column means (center columns)
    D = D - D.mean(axis=0)[None, :]  
    
    # Subtract row means (center rows)
    D = D - D.mean(axis=1)[:, None]   
    
    # Apply PCA to the centered distance matrix
    pca = decomposition.PCA(
        n_components=n_components,
        svd_solver="randomized", 
        random_state=random_state  
    )
    
    # Compute and return the embedding
    Y = pca.fit_transform(D)
    
    return Y


def compute_rank_matrix(K, l1, l2, mode,beta, epsilon, random_state, verbose=0):
    """Compute the rank matrix R through gradient descent optimization.
    
    Args:
        K: Input kernel matrix
        l: Regularization parameter (if None, will be calculated)
        l2: Secondary regularization parameter (if None, will be set to l+1)
        beta: Learning rate for gradient descent
        epsilon: Convergence threshold
        random_state: Random seed for reproducibility
        verbose: Control output verbosity (0 = silent, >0 = show progress)
        
    Returns:
        R: The computed rank matrix
    """
    

    # Compute eigendecomposition of kernel matrix
    if mode == 1:
        Phi,lamb,Psi = eigen_kernel(K)
    elif mode == 2:
        Phi,lamb,Psi = eigen_kernel2(K)

    if l1 is None:
        l1 = calc_l(lamb)
    if l2 is None:
        l2 = l1 + 1
    Lamb_l = np.power(lamb,l2)
    A = Phi * Lamb_l @ Psi
    

    n_samples = K.shape[0]
    # Initialize control variables (cv) with random uniform values
    cv = stats.uniform.rvs(loc=0.5, scale=0.2, size=n_samples, random_state=random_state).round(2)
    
    # Gradient descent iteration
    j = 0    
    while True:
        j = j + 1
        # Compute current rank matrix and its differential
        R, dif_R = compute_infty_R(Phi, lamb, Psi, cv, l1)
        
        # Prepare Q matrix for gradient calculation (replace 0 with inf to avoid division by 0)
        Q = R.copy()
        Q[Q==0] = np.inf
        
        # Compute gradient and update control variables
        Dif = - A/Q * dif_R    
        dif_v = Dif.sum(axis=1) 
        old_cv = cv
        cv = old_cv - beta * dif_v
        
        # Handle control variables that exceed bounds
        count_high_cv = (cv > 1).sum() 
        count_low_cv = (cv < 0).sum()
        count_cv = count_high_cv + count_low_cv
        
        if count_cv > 0:
            if count_high_cv > 0:
                if verbose > 0:
                    warnings.warn('Warning Message: there are %d samples with cv > 0.99' % (count_high_cv))
                    cv[cv>1] = 0.99
            if count_low_cv > 0:
                if verbose > 0:
                    warnings.warn('Warning Message: there are %d samples with cv < 0.01' % (count_low_cv))
                    cv[cv<0] = 0.01
                                
        # Check convergence
        diff = cv - old_cv 
        if verbose > 0:
            if j % 10 == 0:
                print("The number of iterations of gradient descent method is:", j,
                    "the average cv is ", np.mean(cv), " and the max diff is", max(np.abs(diff)))

        if (np.abs(diff) < epsilon).all():
            break

        # Safety stop after 30 iterations
        if j > 30:
            break        
            
    return R