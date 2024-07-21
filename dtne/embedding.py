import numpy as np
import scipy as sp
import pandas as pd
import warnings
import numpy.core.numeric as nx

from scipy import spatial, linalg, sparse, stats
from sklearn import cluster, decomposition, manifold, preprocessing
from sklearn.metrics.pairwise import pairwise_distances

warnings.simplefilter(action='ignore', category=UserWarning)


from .utils import *


class DTNE(object):
    """
    """

    def __init__(self, n_components= 2,  verbose=0, **kwargs):
        """
        Initializes the class with default parameters.
        Args:
            n_components (int, optional): The number of latent components. Defaults to 2.
            verbose (int, optional): Level of verbosity. Defaults to 0.
            **kwargs: Additional keyword arguments that can be used to override default parameters.
        """

        self.n_components = n_components
        self.verbose = verbose
        self.X = None
        self.n_samples = None
        self.n_features = None
        self.delta = None
        self.P = None
        self.Pnm = None
        self.R = None
        self.alpha = None
        self.beta = None
        self.epsilon = None
        self.adjacency = None
        self.Y_ = None
        self.iter_mode = None
        self.random_state = None
        self.cv = None
        self.n_landmark = None
        self.G = None
        self.dists = None
        self.root_cells = None
        self.n_dims = None
        self.solver = None
        self.l1 = None
        self.l2 = None
        self.mode = None
        self.adjacency_knn_indices = None
        self.adjacency_kernel = None
        self.cluster_labels = None
        self.min_dist = None
        self.mark = 0 
        

        self.__set_params__()

        if "k_neighbors" in kwargs:          # number of neighbors for kNN graph
            self.k_neighbors = kwargs["k_neighbors"]
            
        if "delta" in kwargs:                # the parameter for cKNN graph
            self.delta = kwargs["delta"]

        if "include_self" in kwargs:         # whether to include self-loops in adjacency matrix
            self.include_self = kwargs["include_self"]

        if "fixed_d" in kwargs:              # parameter for density of weights.
            self.fixed_d = kwargs["fixed_d"]

        if "alpha" in kwargs:                # The power of the kernel matrix.
            self.alpha = kwargs["alpha"]

        if "epsilon" in kwargs:              # difference threshold parameter for optimal restart probability
            self.epsilon = kwargs["epsilon"]

        if "beta" in kwargs:                 # learning rate for loss function
            self.beta = kwargs["beta"]
        
        if "kernel" in kwargs:               # kernel of the Markov matrix
            self.kernel = kwargs["kernel"]
  
        if "iter_mode" in kwargs:            # mode of diffusion iteration. 
            self.iter_mode = kwargs["iter_mode"]
        
        if "random_state" in kwargs:         # random seed  
            self.random_state = kwargs["random_state"]
        
        if "n_landmark" in kwargs:           # number of landmarks
            self.n_landmark = kwargs["n_landmark"]

        if "l1" in kwargs:                   # initial number of iterations for the Markov matrix
            self.l1 = kwargs["l1"]
        if "l" in kwargs:                    # alias for l1
            self.l1 = kwargs["l"]
        if "l2" in kwargs:                   # initial number of iterations for PPR matrix
            self.l2 = kwargs["l2"]
        if "solver" in kwargs: 
            self.solver = kwargs['solver']
        if "min_dist" in kwargs: 
            self.min_dist = kwargs['min_dist']


    
    def __set_params__(self):
        """
        This function sets the default parameters for the class.
        """

        self.k_neighbors = 15
        self.delta = 1
        self.include_self = True
        self.alpha = 1
        self.epsilon = 1e-2
        self.beta = 0.1
        self.iter_mode = "infty"
        self.random_state = 0
        self.kernel = 'box'
        self.solver = 'mds'
        # self.min_dist = 0.3


        
    def compute_markov_matrix(self, **params):
        """
        Computes the Markov matrix for the given data.

        Args:
            **params: Keyword arguments that can override default parameters.
        Returns:
            The computed Markov matrix.
                    
        """

        # Get data or raise error if not provided.
        if "X" in params:
            self.X = params["X"]
        elif self.X is None:
            raise ValueError("data X must be specified.")
        
        if "delta" in params:             
            self.delta = params["delta"]

        if "include_self" in params:
            self.include_self = params["include_self"]
            
        if "k_neighbors" in params:
            self.k_neighbors = params["k_neighbors"]

        if "n_dims" in params:
            self.n_dims = params["n_dims"]


        self.n_samples, self.n_features = self.X.shape

        # Perform dimensionality reduction with PCA if needed.
        if self.n_dims is not None:
            pca = decomposition.PCA(n_components=self.n_dims,random_state=self.random_state)
            pca_features = pca.fit_transform(self.X)
            self.X = pca_features

        elif self.n_features > 500:
            pca = decomposition.PCA(n_components=100,random_state=self.random_state)
            pca_features = pca.fit_transform(self.X)
            self.X = pca_features  
        
        # Compute the adjacency matrix based on the chosen kernel.
        if self.kernel == 'box':
            adjacency_kernel,self.adjacency_knn_indices = box_kernel(data = self.X, k_neighbors = self.k_neighbors)
        elif self.kernel == 'box2':
            adjacency_kernel,self.adjacency_knn_indices = box_kernel2(data = self.X, k_neighbors = self.k_neighbors,delta=self.delta)
        elif self.kernel == 'gauss':
            adjacency_kernel,self.adjacency_knn_indices = gauss_kernel(data = self.X, k_neighbors = self.k_neighbors, delta=self.delta, alpha=self.alpha)
        elif self.kernel == 'mix':
            adjacency_kernel,self.adjacency_knn_indices = mix_kernel(data = self.X, k_neighbors = self.k_neighbors, delta=self.delta, alpha=self.alpha)
        elif self.kernel == 'umap':
            adjacency_kernel = scanpy_kernel(data = self.X, knn = self.k_neighbors, method='umap')
        elif self.kernel == 'rapids':
            adjacency_kernel = scanpy_kernel(data = self.X, knn = self.k_neighbors, method='rapids')
        elif self.kernel == 'scanpy_gauss':
            adjacency_kernel = scanpy_kernel(data = self.X, knn = self.k_neighbors, method='gauss')
        elif self.kernel == 'phate':
            adjacency_kernel = phate_kernel(data = self.X, knn = self.k_neighbors)
        
        # Convert the adjacency matrix to a dense array if necessary.
        if self.n_landmark is None or self.n_landmark == self.n_samples:
            adjacency_kernel = adjacency_kernel.toarray()

        # Store the computed adjacency matrix and return it.
        self.adjacency_kernel = adjacency_kernel

        return adjacency_kernel
        
    def learn_vectors(self, **params):
        """
        Learns the transformed Markov matrix(R).

        Args:
        **params: Keyword arguments to override default parameters.
        Returns:
        The transformed Markov matrix (R) and damping factor vector (cv).

        """

        # Get parameters from input or use defaults.
        if "alpha" in params:
            alpha = params["alpha"]
        else:
            alpha = self.alpha
        if "X" in params:
            self.X = params["X"]
        elif self.X is None:
            raise ValueError("data X must be specified.")
        if "epsilon" in params:
            epsilon = params["epsilon"]
        else:
            epsilon = self.epsilon
        if "beta" in params:
            beta = params["beta"]
        else:
            beta = self.beta
        if "iter_mode" in params:
            iter_mode = params["iter_mode"]
        else:
            iter_mode = self.iter_mode

        if "l1" in params:                   
            self.l1 = params["l1"]
        if "l" in params:                    # alias for l1
            self.l1 = params["l"]
        if "l2" in params:                   
            self.l2 = params["l2"]

        # Preprocess data (add landmarks) for large datasets:
        self.n_samples, self.n_features = self.X.shape

        if self.n_samples >= 5000 and self.n_samples < 10000 and self.n_landmark is None:
            self.n_landmark = 1000

        elif self.n_samples >= 10000 and self.n_landmark is None:
            self.n_landmark = 2000


        # Compute Markov matrix (adjacency kernel):
        adjacency_kernel = self.compute_markov_matrix(**params)
        P = preprocessing.normalize(adjacency_kernel, norm="l1", axis=1)


        if "n_landmark" in params:
            self.n_landmark = params["n_landmark"]
        
        if self.n_landmark is None or self.n_landmark == self.n_samples:
            n_samples = self.n_samples
            self.mode = 1

        else:
            n_landmark = self.n_landmark
            svd_cluster = cluster.AgglomerativeClustering(n_clusters=n_landmark,connectivity = adjacency_kernel)
            self.cluster_labels = svd_cluster.fit_predict(P @ self.X)

            Pmm,Pnm = compute_landmark_operator(adjacency_kernel,self.cluster_labels,random_state = self.random_state)
            P = Pmm 
            self.Pnm = Pnm
            n_samples = n_landmark
            self.mode = 2
        
        # Initialize cv (damping factor vector).
        if "cv"  in params:
            cv = params["cv"]
        elif self.cv == None:
            cv = stats.uniform.rvs(loc=0.5, scale=0.2, size=n_samples, random_state= self.random_state).round(2)

        if iter_mode == "infty"  and self.mode == 1:
            Phi,lamb,Psi = eigen_kernel(self.adjacency_kernel)
            if self.l1 is None:
                self.l1 = calc_l(lamb)
            if self.l2 is None:
                self.l2 = self.l1 + 1
            Lamb_l = np.power(lamb,self.l2)
            A = Phi * Lamb_l @ Psi

        if iter_mode == "infty"  and self.mode == 2:
            Phi,lamb,Psi = eigen_kernel2(P)
            if self.l1 is None:
                self.l1 = calc_l(lamb)
            if self.l2 is None:
                self.l2 = self.l1 + 1
            Lamb_l = np.power(lamb,self.l2)
            A = Phi * Lamb_l @ Psi

        # Iterate to learn cv and construct R.
        j  =  0
        while True:
            j = j  + 1

            if iter_mode == "infty":
                R,dif_R = compute_infty_R2(Phi,lamb,Psi,cv,self.l1)
                
            Q = R.copy()
            Q[Q==0] = np.inf

            Dif = - A/Q * dif_R
            dif_v = Dif.sum(axis=1) 
            old_cv  = cv
            cv = old_cv - beta * dif_v

            count_high_cv = (cv>1).sum() 
            count_low_cv = (cv<0).sum()
            count_cv = count_high_cv + count_low_cv
            if count_cv > 0:
                if count_high_cv > 0:
                    if self.verbose > 0:
                        warnings.warn('Warning Message: there are %d samples with cv > 0.99' % (count_high_cv))
                    cv[cv>1] = 0.99
                if  count_low_cv > 0:
                    if self.verbose > 0:
                        warnings.warn('Warning Message: there are %d samples with cv < 0.01' % (count_low_cv))
                    cv[cv<0] = 0.01
                if j > 30:
                    break
                                
            diff = cv - old_cv 
            if self.verbose > 0:
                if j % 10 == 0:
                    print("The number of iterations of gradient descent method is:",j,
                      "the average cv is ",np.mean(cv), " and the max diff is", max(np.abs(diff)))

            if (np.abs(diff)<epsilon).all():
                break

            if j > 30:
                break        
            
        self.cv = cv
        self.R = R

        return R,cv
    


    def compute_embedding(self, **params):
        """
        Computes the diffusion embedding of the data.

        Args:
            **params: Keyword arguments that can be used to override default parameters.
        Returns:
            The diffusion embedding(manifold distance matrix) of the data.

        """


        R,cv = self.learn_vectors(**params)
        P = preprocessing.normalize(R, norm="l1", axis=1)
        A = np.sqrt(R)
        G = A @ A.T
        
        H = -2 * np.log(G)
        H[H<0] = 0
        
        np.fill_diagonal(H,0)
        self.G = G
        self.dists = H
        return self.dists
    
    def order_cells(self, **params):
        """
        Orders cells based on their distances to other cells.

        Args:
            params (dict): A dictionary containing parameters.
                - root_cells (list): A list of root cell indices.

        Returns:
            np.ndarray: A numpy array storing the ordered cell distances.

        Raises:
            ValueError: If root_cells is not specified.
        
        """

        if "root_cells" in params:
            self.root_cells = params["root_cells"]
        elif self.root_cells is None:
            raise ValueError("root_cells must be specified.")
         
        root_cells = self.root_cells

        if self.n_landmark is None or self.n_landmark == self.n_samples:
            if len(root_cells) > 1:

                diff_dists = np.sqrt(self.dists[root_cells,:])
                sum_dists = np.sum(diff_dists,axis=0)                
            else:

                sum_dists = np.sqrt(self.dists[root_cells,:])[0]

        else:
            if len(root_cells) == 1:
                root_cells = self.adjacency_knn_indices[root_cells,:3]

            R = self.Pnm @ self.R
            A = np.sqrt(R)
            G = A[root_cells,:] @ A.T

            H = -2 * np.log(G)
            H[H<0] = 0


            diff_dists = np.sqrt(H).sum(axis=0)
            sum_dists = diff_dists.sum(axis=0)

        # # Normalize distances between 0 and 1
        min_dist = np.min(sum_dists)
        max_dist = np.max(sum_dists)
        diff_time = (sum_dists - min_dist)/(max_dist - min_dist)

        self.df_times = diff_time
        return diff_time
    
    def cluster_cells(self,**params):
        """
        Clusters cells into groups based on a precomputed distance matrix.
        Args:
            **params: Keyword arguments that specify clustering parameters.
                * n_clusters (int): The number of clusters to generate. 
                * cluster_method (str, optional): The clustering method to use.  Can be "kmedoids", "agglo" (hierarchical clustering), or "dbscan". Defaults to "agglo".
                * eps (float, optional): The epsilon parameter for DBSCAN clustering. Defaults to 0.5.
                * min_samples (int, optional): The minimum number of samples in a DBSCAN cluster. Defaults to 5.
        Returns:
            A numpy array containing the cluster labels for each cell.
        """

        if "n_clusters" in params:
            n_clusters = params["n_clusters"]
        
        # Perform clustering based on chosen method.
        if "cluster_method" in params:
            cluster_method = params["cluster_method"]
        else:
            cluster_method = "agglo"

        if cluster_method == "kmedoids":
            from sklearn_extra.cluster import KMedoids
            kmedoids_instance = KMedoids(n_clusters=n_clusters, metric='precomputed',random_state=self.random_state).fit(self.dists)
            labels = kmedoids_instance.labels_
        elif cluster_method == "agglo" or cluster_method == "hiera":
            agglo_instance = cluster.AgglomerativeClustering(n_clusters=n_clusters, metric='precomputed',linkage='average').fit(self.dists)
            labels = agglo_instance.labels_

        elif cluster_method == "dbscan":
            if "eps" in params:
                eps = params["eps"]
            else:
                eps = 0.5
            if "min_samples" in params:
                min_samples = params["min_samples"]
            else:
                min_samples = 5

            db_instance = cluster.DBSCAN(eps=eps, min_samples=min_samples,metric='precomputed').fit(self.dists)
            labels = db_instance.labels_
        
        # elif cluster_method == "leiden" or cluster_method == "louvain":
        #     if "resolution" in params:
        #         resolution = params["resolution"]
        #     else:
        #         resolution = 1
        #     if "use_weights" in params:
        #         use_weights = params["use_weights"]
        #     else:
        #         use_weights = True

        #     if "n_neighbors" in params:
        #         n_neighbors = params["n_neighbors"]
        #     else:
        #         n_neighbors = self.k_neighbors

        #     sorted_indices = np.argsort(self.G, axis=1)[:, ::-1]
        #     k_indices = sorted_indices[:, :n_neighbors]
        #     k_similarities = np.take_along_axis(self.G, k_indices, axis=1)

        #     if self.n_landmark is None or self.n_landmark == self.n_samples:
        #         n1 = self.n_samples
        #     else:
        #         n1 = self.n_landmark

        #     indptr = range(0, (n1 + 1) * n_neighbors, n_neighbors)
        #     k_matrix = sparse.csr_matrix((k_similarities.flatten(), k_indices.flatten(), indptr), shape=(n1, n1))
        #     w_matrix = k_matrix.maximum(k_matrix.T) 
        #     w_matrix = w_matrix.astype(np.float32)
        #     w_matrix = preprocessing.normalize(w_matrix, norm="l1", axis=1)

        #     import scanpy as sc
        #     adata = sc.AnnData(X = self.G) # self.G no meaning, just give "n_obs"
        #     adata.uns["neighbors"] = {'connectivities_key': 'connectivities'}
        #     adata.obsp["connectivities"] = w_matrix
            
        #     # U, s, V = sparse.linalg.svds(self.dists, k=10)
        #     # S = s * np.identity(10)
        #     # Y = U @ np.sqrt(S)
        #     # import scanpy as sc
        #     # adata = sc.AnnData(X = Y)
        #     # sc.pp.neighbors(adata,n_neighbors=n_neighbors)

        #     if cluster_method == "leiden":
        #         sc.tl.leiden(adata, resolution = resolution, use_weights = use_weights)
        #         labels = np.array(adata.obs['leiden'].values).astype('int')
        #     elif cluster_method == "louvain":
        #         sc.tl.louvain(adata,resolution = resolution, use_weights = use_weights)
        #         labels = np.array(adata.obs['louvain'].values).astype('int')


        if self.n_landmark is None or self.n_landmark == self.n_samples:
            clusters = labels
        else:
            clusters = np.array([labels[i] for i in self.cluster_labels])

        return clusters
    
    


    def fit(self,  **params):
        """
        Fits the diffusion embedding and computes the low-dimensional visualization of the data.
        Args:
            **params: Keyword arguments that can be used to override default parameters.
                * n_components (int, optional): The number of dimensions for the low-dimensional embedding. Defaults to the value in `self.n_components`.
                * random_state (int, optional): Random seed for the dimensionality reduction algorithm. Defaults to the value in `self.random_state`.
        Returns:
            The object itself (enables method chaining).
        """
        
        if "n_components" in params:
            n_components = params["n_components"]
        else:
            n_components = self.n_components
        if "random_state" in params:
            random_state = params["random_state"]
        else:
            random_state = self.random_state
        
        if "solver" in params: 
            self.solver = params['solver']
        
        if self.solver == 'mds' or self.dists == None:
            dists = self.compute_embedding(**params)
        else:    # Avoiding repeated calculations
            dists = self.dists


        # Perform dimensionality reduction based on the chosen solver.
        if self.solver == 'mds':
            Y_classic = classic(dists, n_components = n_components, random_state = random_state)
            mds = manifold.MDS(n_components = n_components, dissimilarity='precomputed',metric = True,normalized_stress = False,random_state = random_state)
            if self.n_landmark is None or self.n_landmark == self.n_samples:
                self.Y_ = mds.fit_transform(dists,init=Y_classic)
            else:
                Yl =  mds.fit_transform(dists,init=Y_classic)
                self.Y_ = self.Pnm @ Yl
        elif self.solver == 'sgd':
            import s_gd2
            if self.n_landmark is None or self.n_landmark == self.n_samples:
                self.Y_ = s_gd2.mds_direct(self.n_samples, dists, w=None, init=Y_classic, random_seed=random_state)
            else:
                Yl = s_gd2.mds_direct(self.n_landmark, dists, w=None, init=Y_classic, random_seed=random_state)
                self.Y_ = self.Pnm @ Yl
        elif self.solver == 'umap':
            import umap
            if "min_dist" in params: 
                self.min_dist = params['min_dist']
            if self.min_dist == None:
                self.min_dist = 0.3
            if self.n_landmark is None or self.n_landmark == self.n_samples: 
                self.Y_ = umap.UMAP(metric='precomputed',min_dist=self.min_dist,random_state =random_state).fit_transform(dists)
            else:
                Yl = umap.UMAP(metric='precomputed',min_dist=self.min_dist,random_state =random_state).fit_transform(dists)
                self.Y_ = self.Pnm @ Yl


        if "root_cells" in params:
            diff_time = self.order_cells(**params)

        return self
    


    def fit_transform(self, X):
        """
        Fits the diffusion embedding and computes the low-dimensional visualization of the data in a single step.
        Args:
            X (np.ndarray): The data to be embedded.

        Returns:
            np.ndarray: The low-dimensional embedding of the data.
        """

        self.X = X

        self.fit( X = self.X) 

        self.mark = 1 

        return self.Y_