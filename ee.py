import numpy as np
from scipy import spatial,linalg
from sklearn import manifold,decomposition,datasets

# This file details the implementation of the elastic embedding algorithm. The codes originate from the GitHub repository https://github.com/1zb/manifold.


from .utils import *

def calc_affinity(data, sigmas=1, zeroify_diag=True, normalize=True, metric = 'sqeuclidean'):
    """ Calculate affinity matrixs

    Args:
        data: N x D

    Returns:
        euclid_dist: squared euclidean distance N x N
        gauss_aff: gaussian affinities N x N
    """
    euclid_dist = spatial.distance.cdist(data,data,metric=metric)
    gauss_aff = np.exp(-euclid_dist/sigmas)
    if zeroify_diag:
        np.fill_diagonal(euclid_dist, 0)
        np.fill_diagonal(gauss_aff, 0)

    # if normalize:
    #     euclid_dist = euclid_dist / np.sum(euclid_dist)
    #     aff_sum = np.power(gauss_aff.sum(axis=0),-1/2)
    #     aff_sum = np.diag(aff_sum)
    #     gauss_aff = aff_sum @ gauss_aff @ aff_sum

    return (euclid_dist, gauss_aff)


def calc_ee_loss(output_data, attractive_weights, repulsive_weights, lamb):
    """ Calculate the ee loss function

    Args:
        output_data: low-dimensional data N x L
        attractive_weights: matrix N x N
        repulsive_weights: matrix N x N
        lamb: scalar lambda

    Returns:
        value of loss function
    """
    (euclid_dist, gauss_aff) = calc_affinity(output_data, zeroify_diag=False, normalize=False)
    loss = np.sum(attractive_weights * euclid_dist + lamb * repulsive_weights * gauss_aff)

    return (loss, euclid_dist, gauss_aff)


def get_laplacians(weights):
    """ Graph Laplacians

    Args:
        weights: N x N

    Returns:
        laplacian matrix: N x N
        degree matrix: N x N
    """
    degree = np.diag(np.sum(weights, axis=0))
    return (degree - weights)


def ee_linear_search(output_data, attractive_weights, repulsive_weights, lamb, step_size, spectral_direction, gradients, loss):
    """ Backtracking Linear Search

    Args:

    Returns:

    """
    dummy = 0.1 * gradients.ravel().dot(spectral_direction.ravel())
    current_loss, _, gauss_aff = calc_ee_loss(output_data + step_size * spectral_direction, attractive_weights, repulsive_weights, lamb)
    while current_loss > loss + step_size * dummy:
        step_size  = 0.8 * step_size
        current_loss, _, gauss_aff = calc_ee_loss(output_data + step_size * spectral_direction, attractive_weights, repulsive_weights, lamb)

    output_data = output_data + step_size * spectral_direction
    return (output_data, current_loss, gauss_aff, step_size)



def elastic_embedding(attractive_weights, repulsive_weights, n_components=2, lamb=1, step_size=0.1, Y_init=None, num_iters=100, random_state=None, verbose=0):
    """ Elastic Embedding algorithm for dimensionality reduction.

    Args:
        attractive_weights (ndarray): Matrix of attractive weights between data points.
        repulsive_weights (ndarray): Matrix of repulsive weights between data points.
        n_components (int, optional): Number of dimensions for the embedding. Defaults to 2.
        lamb (float, optional): Regularization parameter. Defaults to 1.
        step_size (float, optional): Initial step size for gradient descent. Defaults to 0.1.
        Y_init (ndarray, optional): Initial embedding coordinates. Defaults to None.
        num_iters (int, optional): Maximum number of iterations. Defaults to 100.
        random_state (int, optional): Random seed for reproducibility. Defaults to None.
        verbose (int, optional): Verbosity level (0 = silent, >0 = print progress). Defaults to 0.

    Returns:
        ndarray: Embedded coordinates of shape (n_samples, n_components).
    """
    # Initialize embedding coordinates
    if Y_init is not None:
        Y = Y_init  
    else:
        Y = 1e-5 * np.random.randn(attractive_weights.shape[0], n_components)
    
    loss_value, _, gauss_aff = calc_ee_loss(Y, attractive_weights, repulsive_weights, lamb)
    attractive_laplacians = get_laplacians(attractive_weights)
    attractive_upper = np.linalg.cholesky(attractive_laplacians + 1e-10 * np.eye(attractive_laplacians.shape[0])).T
    
    # Main optimization loop
    for k in range(num_iters):
        Y_old = Y  
       
        repulsive_laplacians = get_laplacians(repulsive_weights * gauss_aff)
        
        gradients = 4 * (attractive_laplacians - repulsive_laplacians).dot(Y_old)
        
        spectral_direction = -np.linalg.solve(attractive_upper, np.linalg.solve(attractive_upper.T, gradients))

        Y, loss_value, gauss_aff, step_size = ee_linear_search(Y_old, attractive_weights, repulsive_weights, lamb, step_size, spectral_direction, gradients, loss_value)
        
        step_value = linalg.norm(step_size * spectral_direction, 'fro') 
    
        if verbose > 0:
            print(k, loss_value, step_size, step_value)
        if step_value < 1e-2:
            break

    return Y  # Return final embedded coordinates


if __name__ == "__main__":
    n_samples = 2000
    X, color = datasets.make_swiss_roll(n_samples, noise=0.1,random_state=0)

    n_neighbors = 8
    n_components = 2
    
    isomap_op = manifold.Isomap(n_neighbors=n_neighbors, n_components=n_components)
    isomap_op.fit_transform(X)
    X_distances = isomap_op.dist_matrix_
    X_distances = X_distances **2
    X_similarity = np.exp(-X_distances)

    Y_init = classic(X_distances,n_components=2)

    Y = elastic_embedding(X_similarity,X_distances,Y_init = Y_init, lamb=1,num_iters=30,verbose=0)

    import matplotlib.pyplot as plt
    plt.figure()
    plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral)
    plt.show()