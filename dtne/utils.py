import numpy as np
import scipy as sp
import pandas as pd
import sys

from scipy import spatial, linalg, sparse,stats,optimize
from sklearn import decomposition,cluster,neighbors,preprocessing
from sklearn.metrics.pairwise import pairwise_distances
# from pynndescent import NNDescent

import warnings
warnings.simplefilter('ignore',sparse.SparseEfficiencyWarning)


def epanechnikov(d, h=1):
    """
    Computes the Epanechnikov kernel, a popular kernel function used for density estimation.

    Args:
        d (np.ndarray or float): An array of distances or a single distance value.
        h (float): The bandwidth parameter, controlling the kernel's width. Defaults to 1.

    Returns:
        np.ndarray or float: The kernel values where distances are within the bandwidth,
        zero for distances outside the bandwidth.

    """
    return np.where(np.abs(d) <= h, 3 / (4 * h) * (1 - np.power(d / h, 2)), 0)

def gauss(d, local_sigma, alpha=1):
    """
    Computes a Gaussian kernel.

    Args:
        d (float or np.ndarray): The distance(s) for which to compute the Gaussian kernel.
        local_sigma (float): The scale (sigma) parameter for the Gaussian function.
        alpha (float, optional): Controls the sharpness of the kernel. Defaults to 1.

    Returns:
        float or np.ndarray: The Gaussian kernel values corresponding to the input distances.
    """

    return np.exp(-np.power(d / local_sigma, alpha))

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

    return kernel_tilde,knn_indices

def box_kernel(data, k_neighbors):
    """    
    Computes a box kernel matrix for the given data using the k-nearest neighbors.
    In this kernel, all distances to the k-nearest neighbors are assigned a value of 1, 
    meaning that each data point is equally weighted within its neighborhood. And then normalize the kernel matrix.

    Args:
        data (np.ndarray): 
            The input data matrix with shape (n_samples, n_features).
        k_neighbors (int): 
            The number of nearest neighbors to consider for each data point.

    Returns:
        tuple:
        - kernel_tilde (scipy.sparse.csr_matrix): 
            The normalized box kernel matrix, where connections between neighbors are weighted by degree normalization.
        - knn_indices (np.ndarray): 
            The indices of the k nearest neighbors for each data point.

    """

    n_samples = data.shape[0] 
    nbrs = neighbors.NearestNeighbors(n_neighbors = k_neighbors, metric='euclidean',n_jobs = -2).fit(data)
    knn_dists,knn_indices = nbrs.kneighbors(data)
    
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

    return kernel_tilde,knn_indices


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
    sigmas = np.sqrt(knn_dists[:, k_neighbors-1])

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

    return kernel_tilde,knn_indices

def gauss_kernel(data, k_neighbors, delta=1, alpha=1):
    """
    Compute a Gaussian kernel matrix.

    Args:
        data: array-like, shape (n_samples, n_features)
            The input data points.
        k_neighbors: int
            The number of nearest neighbors to consider for each data point.
        delta: float, optional (default=1)
            A scaling factor for local sigma computation that controls the bandwidth of the Gaussian kernel.
        alpha: float, optional (default=1)
            A parameter controlling the decay of the Gaussian function.

    Returns:
        tuple:
        - kernel_tilde (scipy.sparse.csr_matrix): The normalized Gaussian kernel matrix.
        - knn_indices (np.ndarray): The indices of the k nearest neighbors for each data point.
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
        kernel[i, :] = gauss(knn_dists[i, :], local_sigmas[i], alpha) # gauss_decay

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

    return kernel_tilde,knn_indices



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
    W = adata.obsp["connectivities"]
    K = graphtools.matrix.set_diagonal(W, 1)
    return K


def phate_kernel(data, knn = 5, decay = 40.0, anisotropy = 0, n_pca= None, **kwargs):
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

    G = graphtools.Graph(data,knn=knn,decay=decay,anisotropy=anisotropy,n_pca=n_pca,use_pygsp=True,random_state=0)
    K = G.kernel
    return K

def calc_l(lamb):
    """
    This function calculates the value of 'l' based on the entropy and its derivatives of a power series of lambda.

    Args:
        lamb: A float value representing the lambda parameter.

    Returns:
        An integer value representing the calculated 'l'.
    """
    dse_list = []
    for i in range(30):
        dse = stats.entropy(np.power(lamb,i))
        dse_list.append(dse)  
    da = np.gradient(dse_list)
    dda = np.gradient(da)
    if np.sum(np.diff(np.sign(dda))) == 0:
        l = 2
    else:
        l = np.where(np.diff(np.sign(dda))!= 0)[0][0] + 2
    return l

    

def eigen_kernel(kernel):
    """
    This function computes the eigenvalues, eigenvectors, and pseudoinverse of a kernel matrix.

    Args:
        kernel: A 2D numpy array representing the kernel matrix.

    Returns:
        A tuple containing three elements:
            - Phi: A 2D numpy array containing the eigenvectors of the kernel matrix (one eigenvector per column).
            - lamb: A 1D numpy array containing the eigenvalues of the kernel matrix (absolute values).
            - Psi: A 2D numpy array containing the pseudoinverse of the eigenvector matrix (Phi).
    """ 
    kernel_sum = kernel.sum(axis=0)
    kd = np.sqrt(kernel_sum)
    ks = np.diag(1/kd)
    Mp = ks @ kernel @ ks
    [lamb,u] = linalg.eigh(Mp)
    
    idx = lamb.argsort()[::-1]   
    lamb = lamb[idx]
    u = u[:,idx]

    v = u.copy()
    v[:,lamb<0] = -u[:,lamb<0]
    lamb = abs(lamb)
    Phi = ks @ u
    Psi = v.T @ np.diag(kd) 

    return Phi,lamb,Psi


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

def compute_landmark_operator(K,labels, random_state = None):
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


def compute_infty_R(Phi,lamb,Psi,cv,l):
    """
    Compute the rank matrix with Eigen_decomposition when the number of iterations tends to infinity 
    when the number of iterations tends to infinity.
    
    Args:
        Phi: array-like, shape (n_samples, n_samples)
            The eigenvectors of the kernel matrix (from the eigen_kernel function).
        lamb: array-like, shape (n_samples)
            The eigenvalues of the kernel matrix (from the eigen_kernel function).
        Psi: array-like, shape (n_samples, n_samples)
            The pseudoinverse of the eigenvector matrix.
        cv: array-like, shape (n_samples)
            A vector representing the coefficient values for each sample.
        l: int
            The power parameter for lambda in the power series.
    Returns:
        A tuple containing two elements:
        R: array-like, shape (n_samples, n_samples)
            The computed rank matrix at the limit where iterations tend to infinity.
        dif_R: array-like, shape (n_samples, n_samples)
            The differential of the rank matrix.
    """

    n_samples = len(cv)

    lamb_l = np.power(lamb,l)
    ncl = np.outer(1-cv,lamb_l)
    dcl = 1-np.outer(cv,lamb)
    

    Sigma = ncl/dcl
    R = Phi * Sigma @ Psi 

    dd_f = np.power(dcl, 2)
    lam_f = lamb_l * (lamb-1)
    nd_f = np.tile(lam_f, (n_samples, 1))
    dSigma = nd_f/dd_f
    dif_R = Phi * dSigma @ Psi 

    R[R<0] = 0
    dif_R[R==0] = 0
    
    R = preprocessing.normalize(R,norm="l1",axis=1)
    return R,dif_R




def classic(D, n_components=2, random_state=None):
    """
    Fast CMDS using random SVD, the codes of this function come from the PHATE algorithm.
    Starting configuration of the embedding to initialize the SMACOF algorithm.

    Parameters
    ----------
    D : array-like, shape=[n_samples, n_samples]
        pairwise distances

    n_components : int, optional (default: 2)
        number of dimensions in which to embed `D`

    random_state : int, RandomState or None, optional (default: None)
        numpy random state

    Returns
    -------
    Y : array-like, embedded data [n_sample, ndim]
    """
    D = D ** 2
    D = D - D.mean(axis=0)[None, :]
    D = D - D.mean(axis=1)[:, None]
    pca = decomposition.PCA(
        n_components=n_components, svd_solver="randomized", random_state=random_state
    )
    Y = pca.fit_transform(D)
    return Y