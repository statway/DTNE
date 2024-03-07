import numpy as np
import scipy as sp
import pandas as pd
import sys

from scipy import spatial, linalg, sparse,stats
from sklearn import decomposition,cluster,neighbors,preprocessing
from sklearn.metrics.pairwise import pairwise_distances
# from pynndescent import NNDescent

import warnings
warnings.simplefilter('ignore',sparse.SparseEfficiencyWarning)


def epanechnikov(d, h=1):
    """
    Epanechnikov kernel.

    Args:
        d: An array of distances.
        h: The bandwidth parameter.

    Returns:
        An array of kernel values.
    """
    return np.where(np.abs(d) <= h, 3 / (4 * h) * (1 - np.power(d / h, 2)), 0)

def gauss(d, local_sigma, alpha=1):
    """
    Calculates the Gaussian kernel.

    Args:
        d (float): The distance value.
        local_sigma (float): The sigma value for the Gaussian function.
        alpha (float, optional): The alpha value for the Gaussian function. Defaults to 1.

    Returns:
        float: The value of the Gaussian function.
    """

    return np.exp(-np.power(d / local_sigma, alpha))

def box(d,local_sigma):
    """
    Compute the box kernel.

    Args:
        d (numpy.ndarray): The input array.
        local_sigma (float): The threshold value.

    Returns:
        numpy.ndarray: The result of applying the box kernel.
    """
    return np.where(d <= local_sigma, 1, 0)


def mix_decay(d,local_sigma,alpha=1):
    """Apply Gaussian decay to the input distance.

    Args:
        d (ndarray): The input distance.
        local_sigma (float): The standard deviation of the Gaussian kernel.
        alpha (float, optional): The decay parameter. Defaults to 1.

    Returns:
        ndarray: The decayed values based on the Gaussian kernel.
    """
    return np.where( d <= local_sigma, 1, np.exp(- np.power(d/local_sigma,alpha)))



def normalized_mix_kernel(data, k_neighbors, delta=1, alpha=1):
    """
    Compute the normalized kernel matrix.

    Parameters:
    - data: array-like, shape (n_samples, n_features)
        The input data.
    - k_neighbors: int
        The number of nearest neighbors to consider.
    - delta: float, optional (default=1)
        Scaling factor for local sigma computation.
    - alpha: float, optional (default=1)
        Parameter for the gauss_decay kernel.

    Returns:
    - norm_kernel: sparse matrix, shape (n_samples, n_samples)
        The normalized kernel matrix.
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

def normalized_box_kernel(data, k_neighbors):

    n_samples = data.shape[0] 
    nbrs = neighbors.NearestNeighbors(n_neighbors = k_neighbors, metric='sqeuclidean',n_jobs = -2).fit(data)
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


def normalized_box_kernel2(data, k_neighbors, delta=1):
    """
    Compute the normalized kernel matrix using the box kernel.

    Parameters:
    - data: numpy array, input data points
    - k_neighbors: int, number of nearest neighbors to consider
    - delta: float, scaling factor for local sigmas
    - alpha: float, scaling factor for the box kernel

    Returns:
    - norm_kernel: sparse matrix, normalized kernel matrix
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

def normalized_gauss_kernel(data, k_neighbors, delta=1, alpha=1):
    """
    Compute the normalized kernel matrix.

    Parameters:
    - data: array-like, shape (n_samples, n_features)
        The input data.
    - k_neighbors: int
        The number of nearest neighbors to consider.
    - delta: float, optional (default=1)
        Scaling factor for local sigma computation.
    - alpha: float, optional (default=1)
        Parameter for the gauss_decay kernel.

    Returns:
    - norm_kernel: sparse matrix, shape (n_samples, n_samples)
        The normalized kernel matrix.
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



def normalized_scanpy_kernel(data, knn=5, method='umap'):
    """
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


def normalized_phate_kernel(data, knn = 5, decay = 40.0, anisotropy = 0, n_pca= None, **kwargs):
    """
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
    Compute the eigenvalues and eigenvectors of a kernel matrix.
    
    Parameters
    ----------
    kernel : array-like, shape=[n_samples, n_samples]

    Returns
    -------
    Phi : array-like, shape=[n_samples, n_samples]
    lamb : array-like, shape=[n_samples]
    Psi : array-like, shape=[n_samples, n_samples]
    """ 
    kernel_sum = kernel.sum(axis=0)
    kd = np.sqrt(kernel_sum)
    ks = np.diag(1/kd)
    Mp = ks @ kernel @ ks
    [lamb,u] = linalg.eigh(Mp)

    v = u.copy()
    v[:,lamb<0] = -u[:,lamb<0]
    lamb = abs(lamb)
    Phi = ks @ u
    Psi = v.T @ np.diag(kd) 

    return Phi,lamb,Psi


def eigen_kernel2(matrix):

    lamb, Phi = np.linalg.eig(matrix)
    Psi = np.linalg.inv(Phi)

    return Phi,lamb,Psi

def compute_landmark_operator(K,n_landmark,labels, random_state = None):

    landmarks = np.unique(labels)
    kmn = sparse.vstack([sparse.csr_matrix(K[labels == i, :].sum(axis=0)) for i in landmarks])

    pmn = preprocessing.normalize(kmn,norm="l1",axis=1)
    pnm = preprocessing.normalize(kmn.T,norm="l1",axis=1)
    pmm = pmn @ pnm
    pmm = pmm.toarray()
    pnm = pnm.toarray()

    return pmm,pnm



def compute_iter_R(P,cv,n_iter):
    """
    Compute the rank matrix after iteration.
    
    Parameters
    ----------
    P : array-like, shape=[n_samples, n_samples]
    cv : array-like, shape=[n_samples]
    n_iter : int
    
    Returns
    -------
    R : array-like, shape=[n_samples, n_samples]
    dif_R : array-like, shape=[n_samples, n_samples]
    """
    
    n_samples = len(cv)
    I = np.identity(n_samples)

    R = P
    DF = P-I
    dif_R = np.zeros((n_samples,n_samples))

    for i in range(n_iter):        # t=i+1
        p_cv = np.power(cv,i)
        dif_cv = (i+1)*p_cv
        p_cv = cv*p_cv
        C_t = np.diag(p_cv)

        DF= P@DF
        dif_C = np.diag(dif_cv)
        R = C_t @ DF + R
        dif_R = dif_C@DF + dif_R
    
    return R,dif_R

def compute_infty_R(matrix,cv,mode):
    """
    Compute the rank matrix with Eigen_decomposition when the number of iterations tends to infinity 
    when the number of iterations tends to infinity.
    
    Parameters
    ----------
    kernel : array-like, shape=[n_samples, n_samples]
    cv : array-like, shape=[n_samples]
    
    Returns
    -------
    R : array-like, shape=[n_samples, n_samples]
    dif_R : array-like, shape=[n_samples, n_samples]
    """

    n_samples = len(cv)

    if mode == 1:
        Phi,lamb,Psi = eigen_kernel(matrix)
    elif mode == 2:
        Phi,lamb,Psi = eigen_kernel2(matrix)

    ncl = np.outer(1-cv,lamb)
    dcl = 1-np.outer(cv,lamb)
    

    Sigma = ncl/dcl
    R = Phi * Sigma @ Psi 

    dd_f = np.power(dcl, 2)
    lam_f = lamb * (lamb-1)
    nd_f = np.tile(lam_f, (n_samples, 1))
    dSigma = nd_f/dd_f
    dif_R = Phi * dSigma @ Psi 

    R[R<0] = 0
    dif_R[R==0] = 0
    
    R = preprocessing.normalize(R,norm="l1",axis=1)

    return R,dif_R

def compute_infty_R2(Phi,lamb,Psi,cv,l = 6):
    """
    Compute the rank matrix with Eigen_decomposition when the number of iterations tends to infinity 
    when the number of iterations tends to infinity.
    
    Parameters
    ----------
    kernel : array-like, shape=[n_samples, n_samples]
    cv : array-like, shape=[n_samples]
    
    Returns
    -------
    R : array-like, shape=[n_samples, n_samples]
    dif_R : array-like, shape=[n_samples, n_samples]
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
    """Fast CMDS using random SVD, the codes of this function come from the PHATE algorithm.

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