import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors
import copy as cp
import subprocess
import os
from neighborfinder import *
import time
from toBareHPF import toBareHPF

class HNC_pcut(BaseEstimator, ClassifierMixin):
    
    ## without confidence label
    ## all labeled samples must be in the source/sink set as given

    def __init__(self, list_lambda, k=15, neighboring=True, 
                 distance_metric='euclidean', weight='RBF', epsilon=1, 
                 add_neighbor=False, adjust_lambda=False, pu_learning=False, debug=False):
        self.epsilon = epsilon
        self.list_lambda = sorted(list_lambda)
        self.k = k
        self.neighboring = neighboring
        self.distance_metric = distance_metric
        self.weight = weight
        self.add_neighbor = add_neighbor
        self.adjust_lambda = adjust_lambda
        self.adjust_weight = None # we use the sum of weights in the graph
        self.neighbors_boo = None
        self.neighbors_dist = None
        self.neighbors_sim = None
        self.neighbors_fitted = False
        self.neighbor_method = "knn"
        self.pu_learning = pu_learning # true if we are solving pu learning problem without negative seeds
        self.debug = debug

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        if self.debug:
            if self.pu_learning:
                print("with only positive samples")
            else:
                print("with both positive and negative samples")
        return self
    
    def fit_neighbor(self, neighbor_boo, neighbor_info, mode="dist"):
        # we only need either neighbor_dist or neighbor_sim
        self.neighbors_boo = neighbor_boo
        self.neighbors_fitted = True
        if mode in {"dist", "distance", "distances"}:
            self.neighbors_dist = neighbor_info
        elif mode in {"sim", "similarity", "similarities"}:
            self.neighbors_sim = neighbor_info
        
        return self
    
    def _get_neighbor(self, data_arr, numneigh=10):

        if self.neighbor_method == "knn":
            boo_arr, dist_arr = get_neighbor_knn(data_arr, k=numneigh, metric=self.distance_metric)
        '''
        elif self.neighbor_method == "sparsecomp":
            boo_arr, dist_arr = get_neighbor_sparsecomp(data_arr, k=numneigh, metric=self.distance_metric)
        elif self.neighbor_method == "hnsw":
            boo_arr, dist_arr = get_neighbor_hnsw(data_arr, k=numneigh, metric=self.distance_metric)
        elif self.neighbor_method == "top":
            boo_arr, dist_arr = get_neighbor_top(data_arr, k=numneigh, metric=self.distance_metric)
        '''
        
        return boo_arr, dist_arr
    
    def _add_neighbor(self, data_arr, boo_arr, dist_arr):

        n_train = self.X_train.shape[0]
        unpaired_indices = [n_train + ind for ind in np.where(np.sum(boo_arr[n_train:,:n_train] | boo_arr[:n_train,n_train:].T, axis=1) == 0)[0]]
        
        if len(unpaired_indices) > 0:
            neigh_added = NearestNeighbors(n_neighbors=1, metric=self.distance_metric)
            neigh_added.fit(data_arr[:n_train,:])
            neighbors_added = neigh_added.kneighbors(data_arr[unpaired_indices,:])

            for i, ind in enumerate(unpaired_indices):
                boo_arr[ind,neighbors_added[1][i,0]] = True
                dist_arr[ind,neighbors_added[1][i,0]] = neighbors_added[0][i,0]

        return boo_arr, dist_arr
    
    
    def _get_weight(self, dist, ref_dist=0):
        # input: distance matrix
        # output: edge weight on the graph
        w_metric = self.weight
        epsilon = self.epsilon
        if w_metric == 'RBF':
            w = np.exp(-(dist**2 / (2 * epsilon ** 2)))
        elif w_metric == 'inverse':
            w = 5/dist
        elif w_metric == 'RBF_norm':
            w = np.exp(-((dist-ref_dist)**2 / (2 * epsilon ** 2)))
        elif w_metric == 'new':
            w = 10*ref_dist/np.maximum(dist, 0.001*ref_dist)
        return w

    def _get_distance(self, X):
        distances= squareform(pdist(X, metric=self.distance_metric))
        return distances
    
    def _construct_graph(self, sim, is_neighbor):

        start_timer = time.perf_counter()
        
        n_train, n_samples = self.X_train.shape[0], sim.shape[0]
        n_test = n_samples - n_train 
        
        list_lambda = self.list_lambda
        if self.adjust_lambda:
            list_lambda /= self.adjust_weight
        num_lambda = len(list_lambda)
        
        source_index, sink_index = n_test+1, n_test+2
        lines = ["","n "+str(int(source_index))+" s", "n "+str(int(sink_index))+" t"]

        # first, get the weights of source and sink adjacent arcs

        if self.debug:
            print("step 1: ", time.perf_counter()-start_timer)
        
        pos_booleans = self.y_train == 1
        idx_positives = np.arange(n_train)[pos_booleans]
        if not self.pu_learning:
            idx_negatives = np.arange(n_train)[np.invert(pos_booleans)]
        idx_unlabeled = np.arange(n_train, n_samples)

        if self.debug:
            print("step 1.1: ", time.perf_counter()-start_timer)
        
        source_weights = 0.5*(sim[idx_unlabeled, :][:, idx_positives].sum(axis=1)+sim[:,idx_unlabeled][idx_positives,:].sum(axis=0))
        if not self.pu_learning:
            sink_weights = sim[idx_unlabeled, :][:, idx_negatives].sum(axis=1)
        else:
            sink_weights = np.zeros(source_weights.shape)

        if self.debug:
            print("step 1.2: ", time.perf_counter()-start_timer)
        # adjust sink/source adjacent weights to optimize runtime of the cut algorithm
        if not self.pu_learning:
            min_weights = np.minimum(source_weights, sink_weights)
            source_weights -= min_weights
            sink_weights -= min_weights
        if self.debug:
            print("step 1.3: ", time.perf_counter()-start_timer)
        # compute the weighted degree, which would be multiplied by lambda values later
        deg_weights = sim[idx_unlabeled, :].sum(axis=1)

        if self.debug:
            print("step 2: ", time.perf_counter()-start_timer)

        for i in range(n_test):
            s_source = "a "+str(int(source_index))+" "+str(int(i+1))
            s_sink  = "a "+str(int(i+1))+" "+str(int(sink_index))
            for l in list_lambda:
                if l > 0:
                    s_source += " "+str(source_weights[i]+(l*deg_weights[i]))
                    s_sink += " "+str(sink_weights[i])
                else:
                    s_source += " "+str(source_weights[i])
                    s_sink += " "+str(sink_weights[i]+((-1)*l*deg_weights[i]))
            lines.append(s_source)
            lines.append(s_sink)

        if self.debug:
            print("step 3: ", time.perf_counter()-start_timer)
        
        if self.neighboring:
            for i in range(n_test-1):
                for j in range(i+1,n_test):
                    if is_neighbor[(i+n_train),(j+n_train)] or is_neighbor[(i+n_train),(j+n_train)]:
                        mutual_weight = 0.5*(sim[i+n_train,j+n_train]+sim[j+n_train,i+n_train])
                        s = "a "+str(int(i+1))+" "+str(int(j+1))+" "+str(mutual_weight)
                        lines.append(s)
                        s = "a "+str(int(j+1))+" "+str(int(i+1))+" "+str(mutual_weight)
                        lines.append(s)
        else:
            for i in range(n_test-1):
                for j in range(i+1,n_test):
                    s = "a "+str(int(i+1))+" "+str(int(j+1))+" "+str(0.5*(sim[i+n_train,j+n_train]+sim[j+n_train,i+n_train]))
                    lines.append(s)
                    s = "a "+str(int(j+1))+" "+str(int(i+1))+" "+str(0.5*(sim[i+n_train,j+n_train]+sim[j+n_train,i+n_train]))
                    lines.append(s)
        
        if self.debug:
            print("step 4: ", time.perf_counter()-start_timer)

        numarcs = len(lines) - 3
        lines[0] = "p par-max "+str(int(n_test+2))+" "+str(int(numarcs))+" "+str(int(num_lambda))
        
        with open("./parametric_cut/parametric_cut_input.txt", "w") as file:
            file.writelines("%s\n" % l for l in lines)
            file.close()
        
        if self.debug:
            print("step 5: ", time.perf_counter()-start_timer)
        
        return lines
    
    def _solve_parametric_cut(self, n_test):
        #subprocess.call(["gcc", "./parametric_cut/pseudopar.c"]) need to run this line only if it is not compiled yet
        subprocess.run(['./parametric_cut/compiled_parametric_cut.out'])
        
        file1 = open('./parametric_cut/parametric_cut_output.txt', 'r')
        lines = file1.readlines()
        file1.close()
        
        pred_arr = np.zeros((n_test,len(self.list_lambda)))
        
        for line in lines[:-2]:
            L = line.split()
            pred_arr[int(L[1])-1,int(L[2])-1:] = 1
            
        os.remove('./parametric_cut/parametric_cut_output.txt')
        
        return pred_arr

    def predict(self, X_test):

        X_train = self.X_train
        n_train, n_test = X_train.shape[0], X_test.shape[0]
        n_samples = n_train+n_test
        X_all = np.concatenate((X_train, X_test), axis=0)
        k = self.k
        if k < 1:
            k = int(k*n_samples)
        
        is_neighbor = self.neighbors_boo
        # we only need either distances or similarities
        distances = self.neighbors_dist 
        similarities = self.neighbors_sim # it would be None if it is not provided along with neighbors_boo in fit_neighbors(), and we would have to compute it
        # note that one of similarities[i,j] and similarities[j,i] could be zero, or both of them could be nonzero and equal to one another

        if similarities is None:
            if self.neighboring:
                if not self.neighbors_fitted:
                    is_neighbor, distances = self._get_neighbor(X_all, numneigh=k)
                if self._add_neighbor:
                    is_neighbor, distances = self._add_neighbor(X_all, is_neighbor, distances)
            else:
                is_neighbor = np.full((X_all.shape[0], X_all.shape[0]), True)
                distances = self._get_distance(X_all)
                
            # adjust distances magnitude with respect to the dimension of the data (by dividing np.sqrt(num features))
            similarities = np.zeros(distances.shape)

            if self.weight=="RBF_norm":
                min_distance = distances[is_neighbor].min()
                similarities[is_neighbor] = self._get_weight(distances[is_neighbor]/np.sqrt(X_train.shape[1]), min_distance)
            else:
                similarities[is_neighbor] = self._get_weight(distances[is_neighbor]/np.sqrt(X_train.shape[1]))
        
        #if self.adjust_lambda:
        #    self.adjust_weight = similarities[np.nonzero(similarities)].sum()

        ##### create text input to parametric cut rather than a networkx graph
        
        lines = self._construct_graph(similarities, is_neighbor)

        # and use the parametric cut rather than hpf
        pred_arr = self._solve_parametric_cut(n_test)
               
        return pred_arr

class HNC_pcut_oneclass_csr(BaseEstimator, ClassifierMixin):
    # always take lambda values that are negative (connect nodes to sink with positive values of lambdas)
    # always have labeled samples for the positive class

    def __init__(self, list_lambda, k=15, neighboring=True, 
                 distance_metric='euclidean', weight='RBF', epsilon=1, 
                 add_neighbor=False, adjust_lambda=False, debug=False):
        self.epsilon = epsilon
        self.list_lambda = sorted(list_lambda)
        self.k = k
        self.neighboring = neighboring
        self.distance_metric = distance_metric
        self.weight = weight
        self.add_neighbor = add_neighbor
        self.neighbors_boo = None
        self.neighbors_dist = None
        self.neighbors_sim = None
        self.neighbors_fitted = False
        self.neighbor_method = "knn"
        self.debug = debug

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        '''
        if self.debug:
            if self.pu_learning:
                print("with only positive samples")
            else:
                print("with both positive and negative samples")
        '''
        return self
    
    def fit_neighbor(self, neighbor_csr):
        self.neighbors_csr = neighbor_csr
        return self
    
    def _get_neighbor(self, data_arr, numneigh=10):

        if self.neighbor_method == "knn":
            boo_arr, dist_arr = get_neighbor_knn(data_arr, k=numneigh, metric=self.distance_metric)
        '''
        elif self.neighbor_method == "hnsw":
            boo_arr, dist_arr = get_neighbor_hnsw(data_arr, k=numneigh, metric=self.distance_metric)
        elif self.neighbor_method == "top":
            boo_arr, dist_arr = get_neighbor_top(data_arr, k=numneigh, metric=self.distance_metric)
        elif self.neighbor_method == "sparsecomp":
            boo_arr, dist_arr = get_neighbor_sparsecomp(data_arr, k=numneigh, metric=self.distance_metric)
        '''
        
        return boo_arr, dist_arr
    
    def _add_neighbor(self, data_arr, boo_arr, dist_arr):

        n_train = self.X_train.shape[0]
        unpaired_indices = [n_train + ind for ind in np.where(np.sum(boo_arr[n_train:,:n_train] | boo_arr[:n_train,n_train:].T, axis=1) == 0)[0]]
        
        if len(unpaired_indices) > 0:
            neigh_added = NearestNeighbors(n_neighbors=1, metric=self.distance_metric)
            neigh_added.fit(data_arr[:n_train,:])
            neighbors_added = neigh_added.kneighbors(data_arr[unpaired_indices,:])

            for i, ind in enumerate(unpaired_indices):
                boo_arr[ind,neighbors_added[1][i,0]] = True
                dist_arr[ind,neighbors_added[1][i,0]] = neighbors_added[0][i,0]

        return boo_arr, dist_arr
    
    
    def _get_weight(self, dist, ref_dist=0):
        # input: distance matrix
        # output: edge weight on the graph
        w_metric = self.weight
        epsilon = self.epsilon
        if w_metric == 'RBF':
            w = np.exp(-(dist**2 / (2 * epsilon ** 2)))
        elif w_metric == 'inverse':
            w = 5/dist
        elif w_metric == 'RBF_norm':
            w = np.exp(-((dist-ref_dist)**2 / (2 * epsilon ** 2)))
        elif w_metric == 'new':
            w = 10*ref_dist/np.maximum(dist, 0.001*ref_dist)
        return w

    def _get_distance(self, X):
        distances= squareform(pdist(X, metric=self.distance_metric))
        return distances
    
    def _create_edge_dict(self, csr_arr):

        n_train, n_samples = self.X_train.shape[0], csr_arr.shape[0]
        n_test = n_samples - n_train 
        
        edge_dict = {}
        source_index, sink_index = n_test, n_test+1

        source_weights = 0.5*(csr_arr[n_train:n_samples, :][:, :n_train].sum(axis=1)+csr_arr[:,n_train:n_samples][:n_train,:].sum(axis=0).T)
        deg_weights = csr_arr[n_train:n_samples, :].sum(axis=1)

        for i in range(n_test):
            edge_dict[(source_index, i)] = (source_weights[i,0],0)
            edge_dict[(i, sink_index)] = (0,(-1)*deg_weights[i,0])

        for row, col in zip(*csr_arr.nonzero()):
            if row >= n_train and col >= n_train and row != col:
                if (row-n_train,col-n_train) in edge_dict:
                    edge_dict[(row-n_train, col-n_train)] = (edge_dict[(row-n_train, col-n_train)][0]+0.5*csr_arr[row, col], 0)
                    edge_dict[(col-n_train, row-n_train)] = (edge_dict[(col-n_train, row-n_train)][0]+0.5*csr_arr[row, col], 0)
                else:
                    edge_dict[(row-n_train, col-n_train)] = (0.5*csr_arr[row, col], 0)
                    edge_dict[(col-n_train, row-n_train)] = (0.5*csr_arr[row, col], 0)

        return edge_dict

    def predict(self, X_test):

        X_train = self.X_train
        n_train, n_test = X_train.shape[0], X_test.shape[0]
        n_samples = n_train+n_test
        
        k = self.k
        if k < 1:
            k = int(k*n_samples)
        
        neighbors_csr = self.neighbors_csr

        # use Alex and Roberto's implementation of parametric cut solver
        edges_dict = self._create_edge_dict(neighbors_csr)
        
        breakpoints = toBareHPF(edges_dict, n_test+2, n_test, n_test+1, self.list_lambda)
        pred_arr = np.zeros((n_test,len(self.list_lambda)))
        for i in range(n_test):
            pred_arr[i,breakpoints[i]-1:] = 1
               
        return pred_arr

class HNC_pcut_2class_csr(BaseEstimator, ClassifierMixin):
    # always take lambda values that are positive (connect nodes to sink with positive values of lambdas)
    # always have labeled samples for the positive class

    def __init__(self, list_lambda, k=15, neighboring=True, 
                 distance_metric='euclidean', weight='RBF', epsilon=1, 
                 add_neighbor=False, adjust_lambda=False, debug=False):
        self.epsilon = epsilon
        self.list_lambda = sorted(list_lambda)
        self.k = k
        self.neighboring = neighboring
        self.distance_metric = distance_metric
        self.weight = weight
        self.add_neighbor = add_neighbor
        self.neighbors_boo = None
        self.neighbors_dist = None
        self.neighbors_sim = None
        self.neighbors_fitted = False
        self.neighbor_method = "knn"
        self.debug = debug

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        '''
        if self.debug:
            if self.pu_learning:
                print("with only positive samples")
            else:
                print("with both positive and negative samples")
        '''
        return self
    
    def fit_neighbor(self, neighbor_csr):
        self.neighbors_csr = neighbor_csr
        return self
    
    def _get_neighbor(self, data_arr, numneigh=10):

        if self.neighbor_method == "knn":
            boo_arr, dist_arr = get_neighbor_knn(data_arr, k=numneigh, metric=self.distance_metric)
        '''
        elif self.neighbor_method == "hnsw":
            boo_arr, dist_arr = get_neighbor_hnsw(data_arr, k=numneigh, metric=self.distance_metric)
        elif self.neighbor_method == "top":
            boo_arr, dist_arr = get_neighbor_top(data_arr, k=numneigh, metric=self.distance_metric)
        elif self.neighbor_method == "sparsecomp":
            boo_arr, dist_arr = get_neighbor_sparsecomp(data_arr, k=numneigh, metric=self.distance_metric)
        '''
        
        return boo_arr, dist_arr
    
    def _add_neighbor(self, data_arr, boo_arr, dist_arr):

        n_train = self.X_train.shape[0]
        unpaired_indices = [n_train + ind for ind in np.where(np.sum(boo_arr[n_train:,:n_train] | boo_arr[:n_train,n_train:].T, axis=1) == 0)[0]]
        
        if len(unpaired_indices) > 0:
            neigh_added = NearestNeighbors(n_neighbors=1, metric=self.distance_metric)
            neigh_added.fit(data_arr[:n_train,:])
            neighbors_added = neigh_added.kneighbors(data_arr[unpaired_indices,:])

            for i, ind in enumerate(unpaired_indices):
                boo_arr[ind,neighbors_added[1][i,0]] = True
                dist_arr[ind,neighbors_added[1][i,0]] = neighbors_added[0][i,0]

        return boo_arr, dist_arr
    
    
    def _get_weight(self, dist, ref_dist=0):
        # input: distance matrix
        # output: edge weight on the graph
        w_metric = self.weight
        epsilon = self.epsilon
        if w_metric == 'RBF':
            w = np.exp(-(dist**2 / (2 * epsilon ** 2)))
        elif w_metric == 'inverse':
            w = 5/dist
        elif w_metric == 'RBF_norm':
            w = np.exp(-((dist-ref_dist)**2 / (2 * epsilon ** 2)))
        elif w_metric == 'new':
            w = 10*ref_dist/np.maximum(dist, 0.001*ref_dist)
        return w

    def _get_distance(self, X):
        distances= squareform(pdist(X, metric=self.distance_metric))
        return distances
    
    def _create_edge_dict(self, csr_arr):

        n_train, n_samples = self.X_train.shape[0], csr_arr.shape[0]
        n_pos = np.sum(self.y_train == 1)
        n_neg = n_train - n_pos
        n_test = n_samples - n_train 
        
        edge_dict = {}
        source_index, sink_index = n_test, n_test+1

        source_weights = 0.5*(csr_arr[n_train:n_samples, :][:, :n_pos].sum(axis=1)+csr_arr[:,n_train:n_samples][:n_pos,:].sum(axis=0).T)
        sink_weights = 0.5*(csr_arr[n_train:n_samples, :][:, n_pos:n_train].sum(axis=1)+csr_arr[:,n_train:n_samples][n_pos:n_train,:].sum(axis=0).T)
        deg_weights = csr_arr[n_train:n_samples, :].sum(axis=1)

        for i in range(n_test):
            if source_weights[i,0] > sink_weights[i,0]:
                edge_dict[(source_index, i)] = (source_weights[i,0]-sink_weights[i,0],0)
            else:
                edge_dict[(i, sink_index)] = (sink_weights[i,0]-source_weights[i,0],0)
            
            edge_dict[(source_index, i)] = (0,deg_weights[i,0])

        for row, col in zip(*csr_arr.nonzero()):
            if row >= n_train and col >= n_train and row != col:
                if (row-n_train,col-n_train) in edge_dict:
                    edge_dict[(row-n_train, col-n_train)] = (edge_dict[(row-n_train, col-n_train)][0]+0.5*csr_arr[row, col], 0)
                    edge_dict[(col-n_train, row-n_train)] = (edge_dict[(col-n_train, row-n_train)][0]+0.5*csr_arr[row, col], 0)
                else:
                    edge_dict[(row-n_train, col-n_train)] = (0.5*csr_arr[row, col], 0)
                    edge_dict[(col-n_train, row-n_train)] = (0.5*csr_arr[row, col], 0)

        return edge_dict

    def predict(self, X_test):

        X_train = self.X_train
        n_train, n_test = X_train.shape[0], X_test.shape[0]
        n_samples = n_train+n_test
        
        k = self.k
        if k < 1:
            k = int(k*n_samples)
        
        neighbors_csr = self.neighbors_csr

        # use Alex and Roberto's implementation of parametric cut solver
        edges_dict = self._create_edge_dict(neighbors_csr)
        
        breakpoints = toBareHPF(edges_dict, n_test+2, n_test, n_test+1, self.list_lambda)
        pred_arr = np.zeros((n_test,len(self.list_lambda)))
        for i in range(n_test):
            pred_arr[i,breakpoints[i]-1:] = 1
               
        return pred_arr

## ===============================================================================================
## ===============================================================================================
## ===============================================================================================

'''
class CHNC_pcut(BaseEstimator, ClassifierMixin):
    
    ## similar to LCSNC except that we are allowed to have both labeled and unlabeled in the training set
    ## mimic the semi-supervised manner of other SSL methods
    
    ## instead of specifying a single value of lambda, we provide a list of lambda
    ## outputs are predictions for all lambda values (shape of predictions = numunlabeled x num lambdas)
    ## self.train_labels also has the shape of numlabeled x num lambdas

    def __init__(self, list_lambda, confidence_coef = 1,
                 k=5, neighboring=True, 
                 distance_metric='euclidean', weight='RBF', epsilon=1, 
                 add_neighbor=False, adjust_confidence=True, percent_edges=0.10,
                 confidence_function='knn', neighbor_method='knn',
                 confidence_weights=None):
        self.epsilon = epsilon
        self.list_lambda = sorted(list_lambda)
        self.confidence_coef = confidence_coef
        self.k = k
        self.neighboring = neighboring #True if we do sparsification, False if we use fully connected graph
        self.distance_metric = distance_metric 
        self.weight = weight # weight function f: f(distance) = similarity
        self.confidence_function = confidence_function # need this if confidence weights are not provided later
        self.confidence_weights = confidence_weights 
        self.add_neighbor = add_neighbor # add neighbor(s) to nodes that are disconnected from the graph
        #self.adjust_lambda = adjust_lambda # we keep it False because lambda does not need adjustment
        #self.adjust_weight = None
        self.adjust_confidence = adjust_confidence

        self.percent_edges = percent_edges

        self.neighbors_boo = None # boolean matrix whether i and j are connected
        self.neighbors_dist = None # pairwise distance matrix
        self.neighbors_sim = None # pairwise similarity matrix
        self.neighbors_fitted = False
        self.neighbor_method = neighbor_method
        self.timer = time.perf_counter()

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train        
        if y_train[-1] >= 0:
            self.numlabeled = len(y_train)
        else:
            self.numlabeled = next(idx for idx,i in enumerate(y_train) if i < 0)
        return self
    
    def fit_confidence(self, c):
        self.confidence_weights = c
        return self

    def fit_neighbor(self, neighbor_boo, neighbor_info, mode="dist"):
        # we only need either neighbor_dist or neighbor_sim
        self.neighbors_boo = neighbor_boo
        self.neighbors_fitted = True
        if mode in {"dist", "distance", "distances"}:
            self.neighbors_dist = neighbor_info
        elif mode in {"sim", "similarity", "similarities"}:
            self.neighbors_sim = neighbor_info
        
        return self
    
    def _get_confidence(self, neigh_arr, distances, similarities=None):
        
        confidence_function = self.confidence_function
        y_train = self.y_train
        n_train = self.X_train.shape[0]
        numlabeled = self.numlabeled
        confidence = np.zeros(numlabeled)
        k = self.k
        
        if confidence_function == 'knn':
            for i in range(numlabeled):
                #neigh_pos = np.mean(y_train[:numlabeled][neigh_arr[i,:numlabeled]])
                #confidence[i] = neigh_pos/k - 0.5
                confidence[i] = np.mean(y_train[:numlabeled][neigh_arr[i,:numlabeled]])
                if y_train[i] == 0:
                    confidence[i] = 1-confidence[i]
                confidence[i] += 0.1
                confidence[i] = 2*(confidence[i]**2)
        elif confidence_function == 'w-knn':
            for i in range(numlabeled):
                neigh_sim = similarities[i,:numlabeled][neigh_arr[i,:numlabeled]]
                pos_neigh_sim = np.dot(y_train[neigh_arr[i,:numlabeled]], neigh_sim)
                confidence[i] = sum(pos_neigh_sim)/sum(neigh_sim) - 0.5
                if y_train[i] == 0:
                    confidence[i] = (-1)*confidence[i]
                confidence[i] = 0.5 + confidence[i] 
        elif confidence_function == "constant":
            confidence = 0.5*np.ones(numlabeled)
        
        return confidence

    def _get_neighbor(self, data_arr, numneigh=10):

        if self.neighbor_method == "knn":
            boo_arr, dist_arr = get_neighbor_knn(data_arr, k=numneigh, metric=self.distance_metric)
        elif self.neighbor_method == "sparsecomp":
            boo_arr, dist_arr = get_neighbor_sparsecomp(data_arr, k=numneigh, metric=self.distance_metric)
        elif self.neighbor_method == "hnsw":
            boo_arr, dist_arr = get_neighbor_hnsw(data_arr, k=numneigh, metric=self.distance_metric)
        elif (self.neighbor_method == "top") or (self.neighbor_method == "top_scipy"):
            boo_arr, dist_arr = get_neighbor_top(data_arr, percent=self.percent_edges)
        
        return boo_arr, dist_arr

    def _add_neighbor(self, data_arr, boo_arr, dist_arr):

        n_train = self.X_train.shape[0]
        unpaired_indices = [n_train + ind for ind in np.where(np.sum(boo_arr[n_train:,:n_train] | boo_arr[:n_train,n_train:].T, axis=1) == 0)[0]]
        
        if len(unpaired_indices) > 0:

            neigh_added = NearestNeighbors(n_neighbors=1, metric=self.distance_metric)
            neigh_added.fit(data_arr[:n_train,:])
            neighbors_added = neigh_added.kneighbors(data_arr[unpaired_indices,:])

            for i, ind in enumerate(unpaired_indices):
                boo_arr[ind,neighbors_added[1][i,0]] = True
                dist_arr[ind,neighbors_added[1][i,0]] = neighbors_added[0][i,0]

        return boo_arr, dist_arr
    
    def _get_distance(self, X):
        distances = squareform(pdist(X, metric=self.distance_metric))
        return distances
    
    def _get_weight(self, dist, ref_dist=0):
        w_metric = self.weight
        epsilon = self.epsilon
        if w_metric == 'RBF':
            w = np.exp(-(dist**2 / (2 * epsilon ** 2)))
        elif w_metric == 'inverse':
            w = 5/dist
        elif w_metric == 'RBF_norm':
            w = np.exp(-((dist-ref_dist)**2 / (2 * epsilon ** 2)))
        elif w_metric == 'new':
            w = 10*ref_dist/np.maximum(dist, 0.001*ref_dist)
        return w
    
    def _construct_graph(self, sim, is_neighbor, conf_w):

        start_timer = time.perf_counter()
        
        n_samples = sim.shape[0]
        n_labeled = self.numlabeled
        y_train = self.y_train
        list_lambda = self.list_lambda
        #if self.adjust_lambda:
        #    list_lambda /= self.adjust_weight
        num_lambda = len(list_lambda)
        
        source_index, sink_index = n_samples+1, n_samples+2

        print("step 1: ", time.perf_counter()-start_timer)
        
        lines = ["","n "+str(int(source_index))+" s", "n "+str(int(sink_index))+" t"]
        for i in range(n_labeled):
            s_source = "a "+str(int(source_index))+" "+str(int(i+1))
            s_sink = "a "+str(int(i+1))+" "+str(int(sink_index))
            if y_train[i] == 1:
                for l in list_lambda:
                    if l > 0:
                        s_source += " "+str(conf_w[i]+l*sim[i,:].sum())
                        s_sink += " 0"
                    else:
                        s_source += " "+str(conf_w[i])
                        s_sink += " "+str((-1)*l*sim[i,:].sum())
            else:
                for l in list_lambda:
                    if l > 0:
                        s_source += " "+str(l*sim[i,:].sum())
                        s_sink += " "+str(conf_w[i])
                    else:
                        s_source += " 0"
                        s_sink += " "+str(conf_w[i]+(-1)*l*sim[i,:].sum())
            lines.append(s_source)
            lines.append(s_sink)
        
        print("step 2: ", time.perf_counter()-start_timer)
        
        lambda_weights = sim[n_labeled:n_samples, :].sum(axis=1)
            
        for i in range(n_labeled, n_samples):
            s_source = "a "+str(int(source_index))+" "+str(int(i+1))
            s_sink  = "a "+str(int(i+1))+" "+str(int(sink_index))
            for l in list_lambda:
                if l > 0:
                    s_source += " "+str(l*lambda_weights[i-n_labeled])
                    s_sink += " 0"
                else:
                    s_source += " 0"
                    s_sink += " "+str(-1*l*lambda_weights[i-n_labeled])
            lines.append(s_source)
            lines.append(s_sink)
        
        print("step 3: ", time.perf_counter()-start_timer)
        
        for i in range(n_samples-1):
            for j in range(i+1, n_samples):
                if is_neighbor[i,j] or is_neighbor[j,i]:
                    mutual_weight = 0.5*(sim[i,j]+sim[j,i])
                    s = "a "+str(int(i+1))+" "+str(int(j+1))+" "+str(mutual_weight)
                    lines.append(s)
                    s = "a "+str(int(j+1))+" "+str(int(i+1))+" "+str(mutual_weight)
                    lines.append(s)
        
        print("step 4: ", time.perf_counter()-start_timer)
        
        numarcs = len(lines) - 3
        print("number of arcs : ", numarcs)
        print("number of nodes : ", n_samples)
        lines[0] = "p par-max "+str(int(n_samples+2))+" "+str(int(numarcs))+" "+str(int(num_lambda))
        
        with open("./parametric_cut/parametric_cut_input.txt", "w") as file:
            file.writelines("%s\n" % l for l in lines)
            file.close()
        
        print("step 5: ", time.perf_counter()-start_timer)
        
        return lines
        
    def _solve_cut(self, graph, n_test):
        #cut, partition = nx.minimum_cut(G, n_unlabeled, n_unlabeled + 1)
        breakpoints, cuts, info = pseudoflow.hpf(graph, -1, -2, const_cap="const")
        labels = np.zeros(len(self.y_train)+n_test, dtype=int)
        for i in range(len(self.y_train)+n_test):
            labels[i] = int(cuts[i][0])
        return labels[0:len(self.y_train)], labels[len(self.y_train):]
    
    def _solve_parametric_cut(self, n_test):
        #subprocess.call(["gcc", current_path+"/parametric_cut/pseudopar-zh.c"])
        subprocess.run(['./parametric_cut/compiled_parametric_cut.out'])
        
        n_train = self.numlabeled
        
        file1 = open('./parametric_cut/parametric_cut_output.txt', 'r')
        lines = file1.readlines()
        file1.close()
        
        pred_arr = np.zeros((n_train+n_test,len(self.list_lambda)))
        
        for line in lines[:-2]:
            L = line.split()
            pred_arr[int(L[1])-1,int(L[2])-1:] = 1
        
        os.remove('./parametric_cut/parametric_cut_output.txt')
        
        return pred_arr
            

    def predict(self, X_test):

        X_train = self.X_train
        n_train, n_test = X_train.shape[0], X_test.shape[0]
        n = n_train+n_test
        k = self.k
        if k < 1:
            k = int(k*(n_train+n_test))

        X_all = np.concatenate((X_train, X_test), axis=0)

        is_neighbor = self.neighbors_boo
        # we only need either distances or similarities
        distances = self.neighbors_dist 
        similarities = self.neighbors_sim # it would be None if it is not provided along with neighbors_boo in fit_neighbors(), and we would have to compute it
        # note that one of similarities[i,j] and similarities[j,i] could be zero, or both of them could be nonzero and equal to one another

        if similarities is None:
            print("start computing similarity")
            if self.neighboring:
                if not self.neighbors_fitted:
                    is_neighbor, distances = self._get_neighbor(X_all, numneigh=k)
                if self.add_neighbor:
                    is_neighbor, distances = self._add_neighbor(X_all, is_neighbor, distances)
            else:
                is_neighbor = np.full((X_all.shape[0], X_all.shape[0]), True)
                distances = self._get_distance(X_all)

            # adjust distances magnitude with respect to the dimension of the data (by dividing np.sqrt(num features))
            similarities = np.full(distances.shape, 0.0)

            if self.weight=="RBF_norm":
                min_distance = distances[is_neighbor].min()
                similarities[is_neighbor] = self._get_weight(distances[is_neighbor]/np.sqrt(X_train.shape[1]), min_distance)
            else:
                similarities[is_neighbor] = self._get_weight(distances[is_neighbor]/np.sqrt(X_train.shape[1]))
        else:
            print("similarity already provided")
            

        if self.confidence_weights is None:
            print("confidence weight not given; computing confidence weight")
            # this is for the case where the confidence weights are not provided, which is the default setting
            self.confidence_weights = self._get_confidence(is_neighbor, distances, similarities)
        else:
            print("confidence weight already provided")
        
        #if self.adjust_lambda:
        #    self.adjust_weight = similarities[np.nonzero(similarities)].sum()
            
        c = self.confidence_weights*self.confidence_coef
        if self.adjust_confidence:
            print("scale by the weighted degree averaged across all nodes")
            if self.neighbor_method == 'knn':
                c *= (k*similarities[np.nonzero(similarities)].mean())
            elif self.neighbor_method == 'top':
                c *= ((n-1)*self.percent_edges*similarities[is_neighbor].mean())
            elif self.neighbor_method == 'top_scipy':
                c *= ((n-1)*self.percent_edges*similarities[is_neighbor].mean())
        
        ##### create text input to parametric cut rather than a networkx graph
        
        print("start constructing graph")
        lines = self._construct_graph(similarities, is_neighbor, c)
        print("parametric cut")
        pred_arr = self._solve_parametric_cut(n_test)
        
        ######################################################################
        
        self.train_labels = pred_arr[:n_train,:] 

        return pred_arr[n_train:,:]

    def _get_train_labels(self):
        return self.train_labels

## ===============================================================================================
## ===============================================================================================
## ===============================================================================================
## ===============================================================================================
## ===============================================================================================
## ===============================================================================================

class CKHNC_pcut(BaseEstimator, ClassifierMixin):
    
    ## similar to LCSNC except that we are allowed to have both labeled and unlabeled in the training set
    ## mimic the semi-supervised manner of other SSL methods
    
    ## instead of specifying a single value of lambda, we provide a list of lambda
    ## outputs are predictions for all lambda values (shape of predictions = numunlabeled x num lambdas)
    ## self.train_labels also has the shape of numlabeled x num lambdas

    def __init__(self, list_lambda, confidence_coef = 1,
                 k=5, k_label=10, neighboring=True, 
                 distance_metric='euclidean', weight='RBF', epsilon=1, 
                 add_neighbor=False, percent_edges=0.10,
                 confidence_function='knn', neighbor_method='knn',
                 confidence_weights=None):
        self.epsilon = epsilon
        self.list_lambda = sorted(list_lambda)
        self.confidence_coef = confidence_coef
        self.k = k
        self.k_label = k_label
        self.neighboring = neighboring #True if we do sparsification, False if we use fully connected graph
        self.distance_metric = distance_metric 
        self.weight = weight # weight function f: f(distance) = similarity
        self.confidence_function = confidence_function # need this if confidence weights are not provided later
        self.confidence_weights = confidence_weights 
        self.add_neighbor = add_neighbor # add neighbor(s) to nodes that are disconnected from the graph
        #self.adjust_lambda = adjust_lambda # we keep it False because lambda does not need adjustment
        #self.adjust_weight = None
        self.adjust_confidence = True 

        self.percent_edges = percent_edges

        self.neighbors_boo = None # boolean matrix whether i and j are connected
        self.neighbors_dist = None # pairwise distance matrix
        self.neighbors_sim = None # pairwise similarity matrix
        self.neighbors_fitted = False
        self.neighbor_method = neighbor_method

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train        
        if y_train[-1] >= 0:
            self.numlabeled = len(y_train)
        else:
            self.numlabeled = next(idx for idx,i in enumerate(y_train) if i < 0)

        return self
    
    def fit_confidence(self, c):
        self.confidence_weights = c
        return self

    def fit_neighbor(self, neighbor_boo, neighbor_info, mode="dist"):
        # we only need either neighbor_dist or neighbor_sim
        self.neighbors_boo = neighbor_boo
        self.neighbors_fitted = True
        if mode in {"dist", "distance", "distances"}:
            self.neighbors_dist = neighbor_info
        elif mode in {"sim", "similarity", "similarities"}:
            self.neighbors_sim = neighbor_info
        
        return self
    
    def _get_confidence(self, neigh_arr, distances, similarities=None):
        
        confidence_function = self.confidence_function
        y_train = self.y_train
        n_train = self.X_train.shape[0]
        numlabeled = self.numlabeled
        confidence = np.zeros(numlabeled)
        k = self.k
        
        if confidence_function == 'knn':
            for i in range(numlabeled):
                #neigh_pos = np.mean(y_train[:numlabeled][neigh_arr[i,:numlabeled]])
                #confidence[i] = neigh_pos/k - 0.5
                confidence[i] = np.mean(y_train[:numlabeled][neigh_arr[i,:numlabeled]])
                if y_train[i] == 0:
                    confidence[i] = 1-confidence[i]
                confidence[i] += 0.1
                confidence[i] = 2*(confidence[i]**2)
        elif confidence_function == 'w-knn':
            for i in range(numlabeled):
                neigh_sim = similarities[i,:numlabeled][neigh_arr[i,:numlabeled]]
                pos_neigh_sim = np.dot(y_train[neigh_arr[i,:numlabeled]], neigh_sim)
                confidence[i] = sum(pos_neigh_sim)/sum(neigh_sim) - 0.5
                if y_train[i] == 0:
                    confidence[i] = (-1)*confidence[i]
                confidence[i] = 0.5 + confidence[i] 
        elif confidence_function == "constant":
            confidence = 0.5*np.ones(numlabeled)
        
        return confidence

    def _get_neighbor(self, data_arr, numneigh=10):

        if self.neighbor_method == "knn":
            boo_arr, dist_arr = get_neighbor_knn(data_arr, k=numneigh, metric=self.distance_metric)
        elif self.neighbor_method == "sparsecomp":
            boo_arr, dist_arr = get_neighbor_sparsecomp(data_arr, k=numneigh, metric=self.distance_metric)
        elif self.neighbor_method == "hnsw":
            boo_arr, dist_arr = get_neighbor_hnsw(data_arr, k=numneigh, metric=self.distance_metric)
        elif (self.neighbor_method == "top") or (self.neighbor_method == "top_scipy"):
            boo_arr, dist_arr = get_neighbor_top(data_arr, percent=self.percent_edges)
        
        return boo_arr, dist_arr

    def _add_neighbor(self, data_arr, boo_arr, dist_arr):

        n_train = self.X_train.shape[0]
        unpaired_indices = [n_train + ind for ind in np.where(np.sum(boo_arr[n_train:,:n_train] | boo_arr[:n_train,n_train:].T, axis=1) == 0)[0]]
        
        if len(unpaired_indices) > 0:

            neigh_added = NearestNeighbors(n_neighbors=1, metric=self.distance_metric)
            neigh_added.fit(data_arr[:n_train,:])
            neighbors_added = neigh_added.kneighbors(data_arr[unpaired_indices,:])

            for i, ind in enumerate(unpaired_indices):
                boo_arr[ind,neighbors_added[1][i,0]] = True
                dist_arr[ind,neighbors_added[1][i,0]] = neighbors_added[0][i,0]

        return boo_arr, dist_arr

    def _get_average_neighbor(self, X_test):

        neigh_arr, _ = get_neighbor_knn(self.X_train, another_data=X_test, k=self.k_label, metric=self.distance_metric)
        y_train = self.y_train
        avg_label = [y_train[neigh_arr[i,:]].mean() for i in range(X_test.shape[0])]

        return avg_label    
    
    def _get_distance(self, X):
        distances = squareform(pdist(X, metric=self.distance_metric))
        return distances
    
    def _get_weight(self, dist, ref_dist=0):
        w_metric = self.weight
        epsilon = self.epsilon
        if w_metric == 'RBF':
            w = np.exp(-(dist**2 / (2 * epsilon ** 2)))
        elif w_metric == 'inverse':
            w = 5/dist
        elif w_metric == 'RBF_norm':
            w = np.exp(-((dist-ref_dist)**2 / (2 * epsilon ** 2)))
        elif w_metric == 'new':
            w = 10*ref_dist/np.maximum(dist, 0.001*ref_dist)
        return w
    
    def _construct_graph(self, sim, is_neighbor, conf_w, avg_label):
        
        n_samples = sim.shape[0]
        n_labeled = self.numlabeled
        y_train = self.y_train
        list_lambda = self.list_lambda
        #if self.adjust_lambda:
        #    list_lambda /= self.adjust_weight
        num_lambda = len(list_lambda)
        
        source_index, sink_index = n_samples+1, n_samples+2
        
        lines = ["","n "+str(int(source_index))+" s", "n "+str(int(sink_index))+" t"]
        for i in range(n_labeled):
            s_source = "a "+str(int(source_index))+" "+str(int(i+1))
            s_sink = "a "+str(int(i+1))+" "+str(int(sink_index))
            if y_train[i] == 1:
                for l in list_lambda:
                    if l > 0:
                        s_source += " "+str(conf_w[i]+l*sim[i,:].sum())
                        s_sink += " 0"
                    else:
                        s_source += " "+str(conf_w[i])
                        s_sink += " "+str((-1)*l*sim[i,:].sum())
            else:
                for l in list_lambda:
                    if l > 0:
                        s_source += " "+str(l*sim[i,:].sum())
                        s_sink += " "+str(conf_w[i])
                    else:
                        s_source += " 0"
                        s_sink += " "+str(conf_w[i]+(-1)*l*sim[i,:].sum())
            lines.append(s_source)
            lines.append(s_sink)
        
        #lambda_weights = sim[n_labeled:n_samples, :].sum(axis=1)
            
        for i in range(n_labeled, n_samples):
            s_source = "a "+str(int(source_index))+" "+str(int(i+1))
            s_sink  = "a "+str(int(i+1))+" "+str(int(sink_index))
            for l in list_lambda:
                if l > 0:
                    s_source += " "+str(l*avg_label[i-n_labeled])
                    s_sink += " 0"
                else:
                    s_source += " 0"
                    s_sink += " "+str(-1*l*(1-avg_label[i-n_labeled]))
            lines.append(s_source)
            lines.append(s_sink)
        
        for i in range(n_samples-1):
            for j in range(i+1, n_samples):
                if is_neighbor[i,j] or is_neighbor[j,i]:
                    mutual_weight = 0.5*(sim[i,j]+sim[j,i])
                    s = "a "+str(int(i+1))+" "+str(int(j+1))+" "+str(mutual_weight)
                    lines.append(s)
                    s = "a "+str(int(j+1))+" "+str(int(i+1))+" "+str(mutual_weight)
                    lines.append(s)
        
        numarcs = len(lines) - 3
        print("number of arcs : ", numarcs)
        print("number of nodes : ", n_samples)
        lines[0] = "p par-max "+str(int(n_samples+2))+" "+str(int(numarcs))+" "+str(int(num_lambda))
        
        with open("./parametric_cut/parametric_cut_input.txt", "w") as file:
            file.writelines("%s\n" % l for l in lines)
            file.close()
        
        return lines
    
    def _solve_parametric_cut(self, n_test):
        #subprocess.call(["gcc", current_path+"/parametric_cut/pseudopar-zh.c"])
        subprocess.run(['./parametric_cut/compiled_parametric_cut.out'])
        
        n_train = self.numlabeled
        
        file1 = open('./parametric_cut/parametric_cut_output.txt', 'r')
        lines = file1.readlines()
        file1.close()
        
        pred_arr = np.zeros((n_train+n_test,len(self.list_lambda)))
        
        for line in lines[:-2]:
            L = line.split()
            pred_arr[int(L[1])-1,int(L[2])-1:] = 1
        
        os.remove('./parametric_cut/parametric_cut_output.txt')
        
        return pred_arr
            

    def predict(self, X_test):

        X_train = self.X_train
        n_train, n_test = X_train.shape[0], X_test.shape[0]
        n = n_train+n_test
        k = self.k
        if k < 1:
            k = int(k*(n_train+n_test))

        X_all = np.concatenate((X_train, X_test), axis=0)

        avg_label = self._get_average_neighbor(X_test)

        is_neighbor = self.neighbors_boo
        # we only need either distances or similarities
        distances = self.neighbors_dist 
        similarities = self.neighbors_sim # it would be None if it is not provided along with neighbors_boo in fit_neighbors(), and we would have to compute it
        # note that one of similarities[i,j] and similarities[j,i] could be zero, or both of them could be nonzero and equal to one another

        if similarities is None:
            if self.neighboring:
                if not self.neighbors_fitted:
                    is_neighbor, distances = self._get_neighbor(X_all, numneigh=k)
                if self.add_neighbor:
                    is_neighbor, distances = self._add_neighbor(X_all, is_neighbor, distances)
            else:
                is_neighbor = np.full((X_all.shape[0], X_all.shape[0]), True)
                distances = self._get_distance(X_all)

            # adjust distances magnitude with respect to the dimension of the data (by dividing np.sqrt(num features))
            similarities = np.full(distances.shape, 0.0)

            if self.weight=="RBF_norm":
                min_distance = distances[is_neighbor].min()
                similarities[is_neighbor] = self._get_weight(distances[is_neighbor]/np.sqrt(X_train.shape[1]), min_distance)
            else:
                similarities[is_neighbor] = self._get_weight(distances[is_neighbor]/np.sqrt(X_train.shape[1]))
            

        if self.confidence_weights is None:
            print("confidence weight not given; computing confidence weight")
            # this is for the case where the confidence weights are not provided, which is the default setting
            self.confidence_weights = self._get_confidence(is_neighbor, distances, similarities)
        
        #if self.adjust_lambda:
        #    self.adjust_weight = similarities[np.nonzero(similarities)].sum()
            
        c = self.confidence_weights*self.confidence_coef
        if self.adjust_confidence:
            # scale by the weighted degree averaged across all nodes
            if self.neighbor_method == 'knn':
                c *= (k*similarities[np.nonzero(similarities)].mean())
            elif self.neighbor_method == 'top':
                c *= ((n-1)*self.percent_edges*similarities[is_neighbor].mean())
            elif self.neighbor_method == 'top_scipy':
                c *= ((n-1)*self.percent_edges*similarities[is_neighbor].mean())
        
        ##### create text input to parametric cut rather than a networkx graph
        
        lines = self._construct_graph(similarities, is_neighbor, c, avg_label)
        pred_arr = self._solve_parametric_cut(n_test)
        
        ######################################################################
        
        self.train_labels = pred_arr[:n_train,:] 

        return pred_arr[n_train:,:]

    def _get_train_labels(self):
        return self.train_labels
'''