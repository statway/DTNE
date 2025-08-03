import numpy as np
import warnings

from scipy import  stats,sparse,spatial,linalg
from sklearn import cluster, decomposition, manifold, preprocessing,neighbors

warnings.simplefilter(action='ignore', category=UserWarning)

from .utils import *
from .ee import *

class DTNE(object):

    def __init__(self,
        n_components: int = 2,
        k_neighbors: int = 15,
        include_self: bool = True,
        kernel: str = 'box',
        random_state: int = 0,
        beta: float = 0.1, 
        epsilon: float = 1e-2,
        verbose: int = 0,
        **kwargs) -> None:

        self.X = None
        self.n_samples = None
        self.n_features = None
        self.Y_ = None
        self.X_pca = None
        self.dists = None
        self.K = None
        self.K2 = None

        self.n_landmark = None
        self.Pnm = None
        self.R = None
        self.l = None
        self.l2 = None
        self.X_labels = None
        self.Ym = None
        self.min_dist = None
        self.sigmas = None
        self.msigmas = None
        self.lmsigmas = None
        self.n_pca = None
        self.root_cells = None
        self.terminal_cells = None
        self.adjacency_knn_indices = None

        
        self.n_components = n_components
        self.k_neighbors = k_neighbors
        self.include_self = include_self
        self.kernel = kernel
        self.random_state = random_state
        self.beta = beta
        self.epsilon = epsilon
        self.verbose = verbose
        
        # self.__set_params__()

        if "delta" in kwargs:                # Scaling factor for kernel.        
            self.delta = kwargs["delta"]
        else:
            self.delta = 2
        if "alpha" in kwargs:                # The decay rate for kernel.  
            self.alpha = kwargs["alpha"]
        else:
            self.alpha = 1

        if "solver" in kwargs: 
            self.solver = kwargs['solver']
        else:
            self.solver = "mds"

        if "n_landmark" in kwargs: 
            self.n_landmark = kwargs['n_landmark']

        if "l1" in kwargs:                   # initial number of iterations for the Markov matrix
            self.l = kwargs["l1"]
        if "l" in kwargs:                    # alias for l1
            self.l = kwargs["l"]
        if "l2" in kwargs:                   # initial number of iterations for PPR matrix
            self.l2 = kwargs["l2"]


    def __set_params__(self) -> None:

        pass
    

    def compute_kernel_matrix(self, data: np.ndarray, **params) -> np.ndarray:
        """Compute the kernel matrix and related quantities for a given dataset.

        Args:
            data (np.ndarray): Input data points of shape (n_samples, n_features).
            k_neighbors (int): Number of nearest neighbors to consider for the kernel.
            **params: Additional parameters, including:
                include_self (bool, optional): Whether to include self-connections in the kernel.
        Returns:
            tuple: Contains the kernel matrix (K), indices of k-nearest neighbors (knn_indices),
                 sigma values (sigmas).
        """
    
        if "include_self" in params:
            self.include_self = params["include_self"]

        if "kernel" in params:
            self.kernel = params["kernel"]
            
        if "k_neighbors" in params:
            self.k_neighbors = params["k_neighbors"]


        # Perform dimensionality reduction with PCA if needed.
        if "n_pca" in params:
            self.n_pca = params["n_pca"]

        if self.n_pca is not None:
            pca = decomposition.PCA(n_components=self.n_pca,random_state=self.random_state)
            self.X_pca  = pca.fit_transform(data)

        elif self.n_features > 500:
            self.n_pca = 100
            pca = decomposition.PCA(n_components=self.n_pca,random_state=self.random_state)
            self.X_pca = pca.fit_transform(data) 

        if self.X_pca is not None:
            data = self.X_pca
        else:
            data = self.X

        # Compute the adjacency matrix based on the chosen kernel.
        if self.kernel == 'box':
            adjacency_kernel,self.adjacency_knn_indices, self.sigmas = box_kernel(data = data, k_neighbors = self.k_neighbors)
        elif self.kernel == 'box2':
            adjacency_kernel,self.adjacency_knn_indices, self.sigmas = box_kernel2(data = data, k_neighbors = self.k_neighbors,delta=self.delta)
        elif self.kernel == 'gauss':
            adjacency_kernel,self.adjacency_knn_indices,self.sigmas = gauss_kernel(data = data, k_neighbors = self.k_neighbors, delta=self.delta, alpha=self.alpha)
        elif self.kernel == 'mix':
            adjacency_kernel,self.adjacency_knn_indices,self.sigmas = mix_kernel(data = data, k_neighbors = self.k_neighbors, delta=self.delta, alpha=self.alpha)
        elif self.kernel == 'umap':
            adjacency_kernel,self.adjacency_knn_indices,self.sigmas = scanpy_kernel(data = data, knn = self.k_neighbors, method='umap')
        elif self.kernel == 'rapids':
            adjacency_kernel,self.adjacency_knn_indices,self.sigmas = scanpy_kernel(data = data, knn = self.k_neighbors, method='rapids')
        elif self.kernel == 'scanpy_gauss':
            adjacency_kernel,self.adjacency_knn_indices,self.sigmas = scanpy_kernel(data = data, knn = self.k_neighbors, method='gauss')
        elif self.kernel == 'phate':
            adjacency_kernel,self.adjacency_knn_indices,self.sigmas = phate_kernel(data = data, knn = self.k_neighbors)

        if self.n_landmark is None or self.n_landmark == self.n_samples:
            adjacency_kernel = adjacency_kernel.toarray()

        self.K = adjacency_kernel

        return self.K, self.sigmas
    

    def learn_vectors(self, **params) -> np.ndarray:
        """Learn low-dimensional vectors using kernel matrix and rank matrix computation.

        Args:
            **params: Additional parameters, including:
                l (float, optional): Parameter for rank matrix computation.
                l2 (float, optional): Secondary parameter for rank matrix computation.
                X (np.ndarray, optional): Input data matrix.
                k_neighbors (int, optional): Number of nearest neighbors for kernel computation.

        Returns:
            np.ndarray: Computed rank matrix (R) representing the learned vectors.
        """

       
        if "l" in params:
            self.l = params["l"]
        if "l2" in params:
            self.l2 = params["l2"]
        if "X" in params:
            self.X = params["X"]
        if "k_neighbors" in params:
            k_neighbors = params["k_neighbors"]
        else:
            k_neighbors = self.k_neighbors
        if "n_landmark" in params:
            self.n_landmark = params["n_landmark"]
        if "kernel" in params:
            kernel = params["kernel"] 
        else:
            kernel = self.kernel

        self.n_samples, self.n_features = self.X.shape


        # Preprocess data (add landmarks) for large datasets:
        if self.n_samples >= 5000 and self.n_samples < 10000 and self.n_landmark is None:
            self.n_landmark = 1000

        elif self.n_samples >= 10000 and self.n_landmark is None:
            self.n_landmark = 2000

        
        if self.K is not None and self.k_neighbors == k_neighbors and self.kernel == kernel:
            K = self.K
        else:
            # Compute Markov matrix (adjacency kernel):
            self.kernel = kernel
            self.K,self.sigmas = self.compute_kernel_matrix(data=self.X,k_neighbors=k_neighbors,**params)
            K = self.K

        if self.n_landmark is None or self.n_landmark == self.n_samples:
            self.mode = 1
        else:
            P = preprocessing.normalize(K, norm="l1", axis=1)
            svd_cluster = cluster.AgglomerativeClustering(n_clusters=self.n_landmark,connectivity = K)
            self.cluster_labels = svd_cluster.fit_predict(P @ self.X)

            Pmm,Pnm = compute_landmark_operator(K,self.cluster_labels,random_state = self.random_state)
            K = Pmm
            self.Pnm = Pnm
            self.mode = 2

        self.R = compute_rank_matrix(
            K=K,
            l1=self.l,
            l2=self.l2,
            mode = self.mode,
            beta= self.beta,
            epsilon = self.epsilon,
            random_state = self.random_state,
            verbose = self.verbose
        )

        return self.R

    
    def compute_dist_matrix(self,**params) -> np.ndarray:
        """Compute a distance matrix based on learned vectors and similarity measures.

        Args:
            **params: Additional parameters passed to the learn_vectors method.

        Returns:
            np.ndarray: Distance matrix (dists) derived from the similarity matrix.
        """
        
        R = self.learn_vectors(**params)

        A = np.sqrt(R)
        self.sims = A @ A.T
        np.fill_diagonal(self.sims,1)
        H = -2 *  np.log(self.sims) 
        H[H<0] = 0
        np.fill_diagonal(H,0)
        self.dists = H

        return self.dists
    

    def reduce_dim(self, **params) -> np.ndarray:

        """Reduce dimensionality of data using specified solver (MDS, UMAP, or Elastic Embedding).

        Args:
            **params: Additional parameters, including:
                n_components (int, optional): Number of dimensions for the output embedding.
                solver (str, optional): Dimensionality reduction method ('mds',  'umap', 'ee').
                min_dist (float, optional): Minimum distance parameter for UMAP.

        Returns:
            np.ndarray: Low-dimensional embedding of shape (n_samples, n_components).
        """

        if "n_components" in params:
            self.n_components = params["n_components"]

        if "solver" in params: 
            self.solver = params['solver']

        if self.solver == 'mds' or self.solver == 'mds2':
            Y_classic = classic(self.dists, n_components = self.n_components, random_state = self.random_state)
            mds = manifold.MDS(n_components = self.n_components, dissimilarity='precomputed',metric = True,normalized_stress = False,random_state = self.random_state)
            
            if self.n_landmark is None or self.n_landmark == self.n_samples:
                self.Y_ = mds.fit_transform(self.dists,init=Y_classic)
            elif self.solver == 'mds':
                self.Ym =  mds.fit_transform(self.dists,init=Y_classic)
                self.Y_ = self.Pnm @ self.Ym

            elif self.solver == 'mds2':
                R2 = self.Pnm @ self.R
                R2[R2<0] = 0
                A = np.sqrt(R2)
                G = A @ A.T
                H = -2 * np.log(G)
                H[H<0] = 0
                np.fill_diagonal(H,0)

                Y_classic = classic(H, n_components = self.n_components, random_state = self.random_state)
                mds = manifold.MDS(n_components = self.n_components, dissimilarity='precomputed',metric = True,normalized_stress = False,random_state = self.random_state)
                self.Y_ = mds.fit_transform(H,init=Y_classic)

        if self.solver == "umap" or self.solver == "UMAP":
            import umap
            if "min_dist" in params: 
                self.min_dist = params['min_dist']
            if self.min_dist == None:
                self.min_dist = 0.3
            if self.n_landmark is None or self.n_landmark == self.n_samples: 
                self.Y_ = umap.UMAP(metric='precomputed',n_components=self.n_components,min_dist=self.min_dist,random_state =self.random_state).fit_transform(self.dists)
            else:
                Yl = umap.UMAP(metric='precomputed',n_components=self.n_components,min_dist=self.min_dist,random_state =self.random_state).fit_transform(self.dists)
                self.Y_ = self.Pnm @ Yl

        if self.solver == 'ee' or self.solver == 'EE':
            Y_classic = classic(self.dists, n_components = self.n_components, random_state = self.random_state)
            if "lamb" in params: 
                lamb = params['lamb']
            else:
                lamb = 1
            if self.n_landmark is None or self.n_landmark == self.n_samples: 
                self.Y_ = elastic_embedding(self.sims,self.dists,self.n_components,Y_init = Y_classic, lamb=lamb,num_iters=100,verbose=self.verbose)
            else:
                self.Ym = elastic_embedding(self.sims,self.dists,self.n_components,Y_init = Y_classic, lamb=lamb,num_iters=100,verbose=self.verbose)
                self.Y_ = self.Pnm @ self.Ym

        
        return self.Y_
    

    def order_cells(self, **params):
        """
        Orders cells based on their distances to other cells in the dataset. This is typically used
        to arrange cells in a sequence according to their manifold distances, which can be useful for 
        trajectory inference or pseudotime analysis.

        Args:
            root_cells (list): A list of root cell indices that serve as starting points for ordering.
            terminal_cells(list): A list of terminal cell indices that serve as end points for ordering.

        Returns:
            np.ndarray: A numpy array containing the normalized distances (diff_time) of each cell  relative to the root cells.

        Raises:
            ValueError: If `root_cells` is not provided in the params.
        
        """

        if "root_cells" in params:
            self.root_cells = params["root_cells"]
        elif self.root_cells is None:
            raise ValueError("root_cells must be specified.")
        
        if "terminal_cells" in params:
            self.terminal_cells = params["terminal_cells"]
         
        # root cells
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
            if len(root_cells) > 1:
                root_cells = np.array(root_cells).reshape(-1,1)
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


        # terminal_cells
        if self.terminal_cells is not None:
            if self.n_landmark is None or self.n_landmark == self.n_samples:
                if len(self.terminal_cells) > 1:
                    termi_dists = np.sqrt(self.dists[self.terminal_cells,:])
                    tsum_dists = np.sum(termi_dists,axis=0)                
                else:
                    tsum_dists = np.sqrt(self.dists[self.terminal_cells,:])[0]
            else:
                if len(self.terminal_cells) == 1:
                    termi_cells = self.adjacency_knn_indices[self.terminal_cells,:3]
                else:
                    termi_cells = np.array(self.terminal_cells).reshape(-1,1)
                G2 = A[termi_cells,:] @ A.T

                H2 = -2 * np.log(G2)
                H2[H2<0] = 0
                termi_dists = np.sqrt(H2).sum(axis=0)
                tsum_dists = termi_dists.sum(axis=0) 
            
            tmin_dist = np.min(tsum_dists)
            tmax_dist = np.max(tsum_dists)
            termi_time = (tsum_dists - tmin_dist)/(tmax_dist - tmin_dist)   

            pt = diff_time/(termi_time + diff_time)
            diff_time = (pt-min(pt))/(max(pt)-min(pt))

        self.df_times = diff_time
        return diff_time
    
    def cluster_cells(self,**params):
        """
        Clusters cells into distinct groups based on the precomputed distance matrix (self.dists).
        This function supports various clustering methods including agglomerative (Hierarchical) clustering, 
        KMedoids, and DBSCAN, allowing flexible clustering of cells depending on the user's needs. 
        
        Args: 
            cluster_method (str, optional): The clustering method to use, and 'agglo' as default clustering methods. Options include:
                * "kmedoids" for KMedoids clustering.
                * "agglo" or "hiera" for Agglomerative Clustering (default: "agglo").
                * "dbscan" for Density-Based Spatial Clustering of Applications with Noise (DBSCAN).
            n_clusters (int): The number of clusters to generate (used for KMedoids and Agglomerative).
            eps (float, optional): The epsilon parameter for DBSCAN clustering. Defaults to 0.5.
            min_samples (int, optional): The minimum number of samples in a DBSCAN cluster. Defaults to 5.
        Returns:
            np.ndarray: An array containing the cluster labels for each cell.
        """
        
        # Perform clustering based on chosen method.
        if "cluster_method" in params:
            cluster_method = params["cluster_method"]
        else:
            cluster_method = "agglo"
        
        if cluster_method == "kmedoids" or cluster_method == "agglo" or cluster_method == "hiera":
            # Get the number of clusters if specified
            if "n_clusters" in params:
                n_clusters = params["n_clusters"]
            else:
                n_clusters = 8

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


        if self.n_landmark is None or self.n_landmark == self.n_samples:
            clusters = labels
        else:
            # If using landmarks, map landmark clusters to the original data points
            clusters = np.array([labels[i] for i in self.cluster_labels])

        return clusters
    

    
    def fit(self, X: np.ndarray,**params) -> None:
        """Fit the model to the input data by computing the distance matrix and reducing dimensionality.

        Args:
            X (np.ndarray): Input data of shape (n_samples, n_features).
            **params: Additional parameters passed to compute_dist_matrix and reduce_dim methods.

        Returns:
            self: The fitted model instance.
        """

        self.X = X
        self.compute_dist_matrix(**params)
        
        self.reduce_dim(**params)
        
        return self
    
    def fit_transform(self,X: np.ndarray) -> np.ndarray:

        """Fit the model to the input data and return the low-dimensional embedding.

        Args:
            X (np.ndarray): Input data of shape (n_samples, n_features).

        Returns:
            np.ndarray: Low-dimensional embedding of shape (n_samples, n_components).
        """

        self.fit(X = X) 
        
        return self.Y_