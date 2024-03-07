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

    def __init__(self, n_components= 2,  verbose=0, **kwargs):

        self.n_components = n_components
        self.verbose = verbose
        self.X = None
        self.n_samples = None
        self.n_features = None
        self.delta = None
        self.P = None
        self.Pnm = None
        self.R = None
        self.beta = None
        self.epsilon = None
        self.adjacency = None
        self.embedding_ = None
        self.iter_mode = None
        self.random_state = None
        self.cv = None
        self.n_landmark = None
        self.dists = None
        self.root_cells = None
        self.df_dists = None
        self.n_dims = None
        self.solver = None
        self.l1 = None
        self.l2 = None
        self.mode = None
        self.adjacency_knn_indices = None
        self.cluster_labels = None
        

        self.__set_params__()

        if "k_neighbors" in kwargs:         # k-nearest neighbors
            self.k_neighbors = kwargs["k_neighbors"]

        if "n_iter" in kwargs:
            self.n_iter = kwargs["n_iter"]
            
        if "delta" in kwargs:                #  the parameter of cKNN
            self.delta = kwargs["delta"]

        if "include_self" in kwargs:         # the parameter for whether the adjacency matrix contains diagonal elements.
            self.include_self = kwargs["include_self"]

        if "fixed_d" in kwargs:              # the parameter about density of weights. 
            self.fixed_d = kwargs["fixed_d"]

        if "alpha" in kwargs:                # the parameter of entropy regularization.
            self.alpha = kwargs["alpha"]

        if "epsilon" in kwargs:              # the diff parameter about optimal restart probability.
            self.epsilon = kwargs["epsilon"]

        if "beta" in kwargs:                # the parameter of learning rate for loss function.
            self.beta = kwargs["beta"]
        
        if "kernel" in kwargs:
            self.kernel = kwargs["kernel"]
  
        if "iter_mode" in kwargs:          # the mode of diffusion iteration. 
            self.iter_mode = kwargs["iter_mode"]
        
        if "random_state" in kwargs:
            self.random_state = kwargs["random_state"]
        
        if "n_landmark" in kwargs:
            self.n_landmark = kwargs["n_landmark"]
        if "l1" in kwargs:
            self.l1 = kwargs["l1"]
        if "l" in kwargs:
            self.l1 = kwargs["l"]
        if "l2" in kwargs:
            self.l2 = kwargs["l2"]

    
    def __set_params__(self):

        self.k_neighbors = 15
        self.n_iter = 20
        self.delta = 1
        self.include_self = True
        self.alpha = 1
        self.epsilon = 1e-2
        self.beta = 0.1
        self.iter_mode = "infty"
        self.random_state = 0
        self.kernel = 'box'
        self.solver = 'mds'

        
    def compute_markov_matrix(self, **params):
        """
        Computes the Markov matrix for the given data.
                    
        """
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

        if self.n_dims is not None:
            pca = decomposition.PCA(n_components=self.n_dims,random_state=self.random_state)
            pca_features = pca.fit_transform(self.X)
            self.X = pca_features

        elif self.n_features > 500:
            pca = decomposition.PCA(n_components=100,random_state=self.random_state)
            pca_features = pca.fit_transform(self.X)
            self.X = pca_features  
        
        if self.kernel == 'box':
            adjacency_kernel,self.adjacency_knn_indices = normalized_box_kernel(data = self.X, k_neighbors = self.k_neighbors)
        elif self.kernel == 'box2':
            adjacency_kernel,self.adjacency_knn_indices = normalized_box_kernel2(data = self.X, k_neighbors = self.k_neighbors,delta=self.delta)
        elif self.kernel == 'gauss':
            adjacency_kernel,self.adjacency_knn_indices = normalized_gauss_kernel(data = self.X, k_neighbors = self.k_neighbors, delta=self.delta, alpha=self.alpha)
        elif self.kernel == 'mix':
            adjacency_kernel,self.adjacency_knn_indices = normalized_mix_kernel(data = self.X, k_neighbors = self.k_neighbors, delta=self.delta, alpha=self.alpha)
        elif self.kernel == 'umap':
            adjacency_kernel = normalized_scanpy_kernel(data = self.X, knn = self.k_neighbors, method='umap')
        elif self.kernel == 'rapids':
            adjacency_kernel = normalized_scanpy_kernel(data = self.X, knn = self.k_neighbors, method='rapids')
        elif self.kernel == 'scanpy_gauss':
            adjacency_kernel = normalized_scanpy_kernel(data = self.X, knn = self.k_neighbors, method='gauss')
        elif self.kernel == 'phate':
            adjacency_kernel = normalized_phate_kernel(data = self.X, knn = self.k_neighbors)
        
        if self.n_landmark is None or self.n_landmark == self.n_samples:
            adjacency_kernel = adjacency_kernel.toarray()
        else:
            adjacency_kernel = adjacency_kernel

        self.adjacency_kernel = adjacency_kernel

        return adjacency_kernel
        
    def learn_vectors(self, **params):

        if "alpha" in params:
            alpha = params["alpha"]
        else:
            alpha = self.alpha
        if "X" in params:
            self.X = params["X"]
        elif self.X is None:
            raise ValueError("data X must be specified.")
        if "n_iter" in params:
            n_iter = params["n_iter"]
        else:
            n_iter = self.n_iter
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
        if "l" in params:
            self.l = params["l"]

        self.n_samples, self.n_features = self.X.shape

        if self.n_samples >= 5000 and self.n_samples < 10000 and self.n_landmark is None:
            self.n_landmark = 1000

        elif self.n_samples >= 10000 and self.n_landmark is None:
            self.n_landmark = 2000

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

            Pmm,Pnm = compute_landmark_operator(adjacency_kernel,n_landmark,self.cluster_labels,random_state = self.random_state)
            P = Pmm 
            self.Pnm = Pnm
            n_samples = n_landmark
            self.mode = 2
        
        
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
            # print("hello",self.l1,self.l2)

        j  =  0
        # A1 = normalized_gauss_kernel(data = self.X, k_neighbors = 2*self.k_neighbors, delta=self.delta, alpha=self.alpha)
        # A1 = normalized_phate_kernel(data = self.X, knn = self.k_neighbors)
        # A = preprocessing.normalize(A1, norm="l1", axis=1)
        # A = P @ P 
        while True:
            j = j  + 1

            if iter_mode == "iter":
                R,dif_R = compute_iter_R(P,cv,n_iter)
            elif iter_mode == "infty":
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
        np.emath.logn

        return R,cv
    

    def compute_embedding(self, **params):


        R,cv = self.learn_vectors(**params)
        P = preprocessing.normalize(R, norm="l1", axis=1)
        A = np.sqrt(R)
        G = A @ A.T
        H = -np.log(G)
        np.fill_diagonal(H,0)
        self.dists = H
        # H = 1 - G
        # np.fill_diagonal(H,0)
        # self.dists = H

        return H
    
    def order_cells(self, **params):

        if "root_cells" in params:
            self.root_cells = params["root_cells"]
        elif self.root_cells is None:
            raise ValueError("root_cells must be specified.")
         
        root_cells = self.root_cells

        if self.n_landmark is None or self.n_landmark == self.n_samples:
            if len(root_cells) > 1:
                diff_dists = np.sqrt(2*self.dists[root_cells,:])
                sum_dists = np.sum(diff_dists,axis=0)                
            else:
                sum_dists = np.sqrt(2*self.dists[root_cells,:])[0]

            min_dist = np.min(sum_dists)
            max_dist = np.max(sum_dists)
            diff_time = (sum_dists - min_dist)/(max_dist - min_dist)

        else:
            if len(root_cells) == 1:
                root_cells = self.adjacency_knn_indices[root_cells,:3]

            R = self.Pnm @ self.R
            A = np.sqrt(R)
            G = A[root_cells,:] @ A.T
            H = -np.log(G)
            H[H<0] = 0

            diff_dists = np.sqrt(2*H).sum(axis=0)
            sum_dists = diff_dists.sum(axis=0)
            # print(diff_dists.shape,sum_dists.shape)
            min_dist = np.min(sum_dists)
            max_dist = np.max(sum_dists)
            diff_time = (sum_dists - min_dist)/(max_dist - min_dist)

        self.df_times = diff_time
        return diff_time
    
    def cluster_cells(self,**params):

        if "n_clusters" in params:
            n_clusters = params["n_clusters"]
        if "dim_y" in params:
            dim_y = params["dim_y"]
        else:
            dim_y = 8
        
        if "cluster_method" in params:
            cluster_method = params["cluster_method"]
        else:
            cluster_method = "agglo"

        if cluster_method == "kmedoids":
            from sklearn_extra.cluster import KMedoids
            kmedoids_instance = KMedoids(n_clusters=n_clusters, metric='precomputed',random_state=self.random_state).fit(self.dists)
            labels = kmedoids_instance.labels_
        elif cluster_method == "agglo":
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
            clusters = np.array([labels[i] for i in self.cluster_labels])

        return clusters
    
    


    def fit(self,  **params):
        
        if "n_components" in params:
            n_components = params["n_components"]
        else:
            n_components = self.n_components
        if "random_state" in params:
            random_state = params["random_state"]
        else:
            random_state = self.random_state

        dists = self.compute_embedding(**params)

        if self.solver == 'mds':
            Y_classic = classic(dists, n_components = n_components, random_state = random_state)
            mds = manifold.MDS(n_components = n_components, dissimilarity='precomputed',metric = True,normalized_stress = False,random_state = random_state)
            if self.n_landmark is None or self.n_landmark == self.n_samples:
                self.embedding_ = mds.fit_transform(dists,init=Y_classic)
            else:
                Yl =  mds.fit_transform(dists,init=Y_classic)
                self.embedding_ = self.Pnm @ Yl
        elif self.solver == 'sgd':
            import s_gd2
            if self.n_landmark is None or self.n_landmark == self.n_samples:
                self.embedding_ = s_gd2.mds_direct(self.n_samples, dists, w=None, init=Y_classic, random_seed=random_state)
            else:
                Yl = s_gd2.mds_direct(self.n_landmark, dists, w=None, init=Y_classic, random_seed=random_state)
                self.embedding_ = self.Pnm @ Yl


        if "root_cells" in params:
            diff_time = self.order_cells(**params)

        return self
    


    def fit_transform(self, X):

        self.X = X

        self.fit( X = self.X)  

        return self.embedding_