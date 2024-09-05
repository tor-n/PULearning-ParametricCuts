import numpy as np
from readData import readData, data_prep, news_data_prep, cifar_data_prep, mnist_data_prep
from neighborfinder import get_neighbor_knn, get_neighbor_csr
from models_parametric_cut_PU import HNC_pcut, HNC_pcut_oneclass_csr, HNC_pcut_2class_csr
from sklearn.metrics import accuracy_score, f1_score
from trees import PUExtraTrees
import sys
import argparse

def process_args(arguments):
    parser = argparse.ArgumentParser(
        description='HNC-PU Learning implementation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--dataset', '-d', default='mnist', type=str,
                        help='The dataset name')
    parser.add_argument('--labeled', '-l', default=100, type=int,
                        help='# of labeled data')
    parser.add_argument('--labeledclass', '-lc', default=1, type=int,
                        help='class of labeled data')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--seed', '-S',default=None, type=int,
                        help='Random seed for splitting dataset into training and test set')
    parser.add_argument('--negativeseed', '-M', default=None, type=int,
                        help='Random seed for sampling reliable negatives')
    parser.add_argument('--priorshift', '-P', default=1., type=float,
                        help='(mis)specification of prior')
    parser.add_argument('--featureweights', '-fi', default=True, action=argparse.BooleanOptionalAction)
    args = parser.parse_args(arguments)

    if args.dataset == "vote":
        args.labeled = 40
    elif args.dataset == "obesity":
        args.labeled = 100
    elif args.dataset == "obesity":
        args.labeled = 400
    elif args.dataset == "news":
        args.labeled = 1000
    elif args.dataset == "letter":
        args.labeled = 1000
    elif args.dataset == "cifar":
        args.labeled = 3600
    elif args.dataset == "mnist":
        args.labeled = 3500
    
    args.negativeseed = args.seed-2515

    return args

def negative_selector(pred_arr, num_selected=10, method="random", random_seed=27):
    # get the indices in each column of example_diff where the entries are 1
    diff_arr = np.diff(pred_arr, axis=1)
    np.random.seed(random_seed)
    most_neg = np.concatenate([np.random.permutation(np.where(diff_arr[:,i] == 1)[0]) for i in range(diff_arr.shape[1])][::-1])
    if method == "ranking":
        # select the top num_selected indices
        return most_neg[:num_selected], most_neg
    elif method == "alternate":
        # select even entries of most_neg
        return most_neg[:2*num_selected][1::2], most_neg
    elif method == "random":
        # select fist several entries with higher chance, last entries with lower chance
        num_consider = min(num_selected*4, len(most_neg))
        selected = np.random.choice(most_neg[:num_consider], num_selected, replace=False, p=np.arange(num_consider,0,-1)/(num_consider*(num_consider+1)/2))
        return selected, most_neg

def pu_hnc(data_X, data_y, labeled_size, num_reliable_negative=50, k_sparsify=None, feature_importance=False, negative_selector_method=["random"], metrics=['acc','f1'], select_neg_seed=27, args_HNC=None):

    # compute feature importance if needed
    # and scale the data using the feature importance
    if feature_importance:
        g = PUExtraTrees()
        g.fit(P=data_X[:labeled_size,:], U=data_X, pi = data_y.mean())
        fi = g.feature_importances()
        data_X = data_X*np.sqrt(data_X.shape[1])*fi
        print("feature importance applied")
    
    # split data into labeled and unlabeled
    X_train, X_test = data_X[:labeled_size,:], data_X[labeled_size:,:]
    y_train, y_test = data_y[:labeled_size], data_y[labeled_size:]
    
    scorers = {'acc': accuracy_score, 'f1': f1_score}
    colors = {'acc': 'hotpink', 'f1': 'darkorange'}

    num_selectors = 1

    if k_sparsify is None:
        k_sparsify = args_HNC['k']
    else:
        args_HNC['k'] = k_sparsify
    
    # sparsify the graph (select pairs to be considered using knn sparsification)
    neighbors_boo, neighbors_dist = get_neighbor_knn(data_X, k=k_sparsify)

    # solve the problem using only positive labeled samples
    # want lambda to be negative for the first step
    if args_HNC['list_lambda'][1] > 0:
        args_HNC['list_lambda'] = sorted([(-1)*lamb for lamb in args_HNC['list_lambda']])
    
    args_HNC['pu_learning'] = True

    hnc = HNC_pcut(**args_HNC)
    hnc.fit(X_train, y_train)
    hnc.fit_neighbor(neighbors_boo, neighbors_dist)

    y_pred = hnc.predict(X_test)

    test_scores = np.zeros((num_selectors, len(metrics), 2*y_pred.shape[1]))
    positive_percentage = np.zeros((num_selectors, 2*y_pred.shape[1]))
    for i in range(y_pred.shape[1]):
        for j, metric in enumerate(metrics):
            test_scores[:,j,i] = scorers[metric](y_test, y_pred[:,i])
        positive_percentage[:,i] = (y_pred[:,i].sum()+labeled_size)/data_X.shape[0]
    
    # now solve the problem using positive and reliable negative labeled samples
    # (change to positive lambdas)
    if args_HNC['list_lambda'][1] < 0:
        args_HNC['list_lambda'] = sorted([(-1)*lamb for lamb in args_HNC['list_lambda']])

    args_HNC['pu_learning'] = False
    
    # reliable negatives
    
    for selector_ind, selector in enumerate(negative_selector_method):
        if selector=="skip":
            continue

        reliable_negatives, _ = negative_selector(y_pred, num_selected=num_reliable_negative, method=selector, random_seed=select_neg_seed)
        
        # construct new labeled-unlabeled sets and preidct
        X_train2 = np.concatenate((X_train, X_test[reliable_negatives,:])) 
        y_train2 = np.concatenate((y_train, np.array([0 for i in range(len(reliable_negatives))])))
        X_test2, y_test2 = np.delete(X_test, reliable_negatives, axis=0), np.delete(y_test, reliable_negatives)

        neighbors_boo, neighbors_dist = get_neighbor_knn(np.concatenate((X_train2, X_test2), axis=0), k=k_sparsify)

        hnc2 = HNC_pcut(**args_HNC)
        hnc2.fit(X_train2, y_train2)

        hnc2.fit_neighbor(neighbors_boo, neighbors_dist)
        y_pred2 = hnc2.predict(X_test2) 

        y_actual_test = np.concatenate((y_test2, y_test[reliable_negatives]))
        y_actual_pred = np.concatenate((y_pred2, np.zeros((len(reliable_negatives), y_pred2.shape[1]))))

        for i in range(y_actual_pred.shape[1]):
            for j, metric in enumerate(metrics):
                test_scores[selector_ind, j ,y_pred.shape[1]+i] = scorers[metric](y_actual_test, y_actual_pred[:,i])
            
            positive_percentage[selector_ind,y_pred.shape[1]+i] = (y_actual_pred[:,i].sum()+labeled_size)/data_X.shape[0]

    # find the index of positive percentage closest to the true positive percentage
    closest_index = np.argmin(np.abs(positive_percentage - data_y.mean()), axis=1)
    closest_percentage = positive_percentage[np.arange(num_selectors), closest_index]
    
    return {"pos": closest_percentage[0], "acc":test_scores[0, 0, closest_index[0]], "f1":test_scores[selector_ind, 1, closest_index[0]]}

def pu_hnc_csr(data_X, data_y, labeled_size, num_reliable_negative=50, k_sparsify=None, feature_importance=False, negative_selector_method=["random"], metrics=['acc','f1'], select_neg_seed=27, args_HNC=None):

    # compute feature importance if needed
    # and scale the data using the feature importance
    if feature_importance:
        g = PUExtraTrees()
        g.fit(P=data_X[:labeled_size,:], U=data_X, pi = data_y.mean())
        fi = g.feature_importances()
        data_X = data_X*np.sqrt(data_X.shape[1])*fi
        print("feature importance applied")
    
    # split data into labeled and unlabeled
    X_train, X_test = data_X[:labeled_size,:], data_X[labeled_size:,:]
    y_train, y_test = data_y[:labeled_size], data_y[labeled_size:]
    
    scorers = {'acc': accuracy_score, 'f1': f1_score}
    colors = {'acc': 'hotpink', 'f1': 'darkorange'}

    num_selectors = 1

    if k_sparsify is None:
        k_sparsify = args_HNC['k']
    else:
        args_HNC['k'] = k_sparsify
    
    # sparsify the graph (select pairs to be considered using knn sparsification)
    #neighbors_boo, neighbors_dist = get_neighbor_knn(data_X, k=k_sparsify)
    neighbors_csr = get_neighbor_csr(data_X, k=k_sparsify, epsilon=args_HNC['epsilon'])

    # solve the problem using only positive labeled samples
    # want lambda to be negative for the first step
    if args_HNC['list_lambda'][1] > 0:
        args_HNC['list_lambda'] = sorted([(-1)*lamb for lamb in args_HNC['list_lambda']])
    
    #args_HNC['pu_learning'] = True

    hnc = HNC_pcut_oneclass_csr(**args_HNC)
    hnc.fit(X_train, y_train)
    hnc.fit_neighbor(neighbors_csr)

    y_pred = hnc.predict(X_test)

    test_scores = np.zeros((num_selectors, len(metrics), 2*y_pred.shape[1]))
    positive_percentage = np.zeros((num_selectors, 2*y_pred.shape[1]))
    for i in range(y_pred.shape[1]):
        for j, metric in enumerate(metrics):
            test_scores[:,j,i] = scorers[metric](y_test, y_pred[:,i])
        positive_percentage[:,i] = (y_pred[:,i].sum()+labeled_size)/data_X.shape[0]
    
    # now solve the problem using positive and reliable negative labeled samples
    # (change to positive lambdas)
    if args_HNC['list_lambda'][1] < 0:
        args_HNC['list_lambda'] = sorted([(-1)*lamb for lamb in args_HNC['list_lambda']])

    #args_HNC['pu_learning'] = False
    
    # reliable negatives
    
    for selector_ind, selector in enumerate(negative_selector_method):
        if selector=="skip":
            continue

        reliable_negatives, _ = negative_selector(y_pred, num_selected=num_reliable_negative, method=selector, random_seed=select_neg_seed)
        
        # construct new labeled-unlabeled sets and preidct
        X_train2 = np.concatenate((X_train, X_test[reliable_negatives,:])) 
        y_train2 = np.concatenate((y_train, np.array([0 for i in range(len(reliable_negatives))])))
        X_test2, y_test2 = np.delete(X_test, reliable_negatives, axis=0), np.delete(y_test, reliable_negatives)

        #neighbors_boo, neighbors_dist = get_neighbor_knn(np.concatenate((X_train2, X_test2), axis=0), k=k_sparsify)
        #neighbors_csr = get_neighbor_csr(np.concatenate((X_train2, X_test2), axis=0), k=k_sparsify, epsilon=args_HNC['epsilon'])

        # reindex
        new_indices = np.concatenate((np.arange(labeled_size), labeled_size+reliable_negatives, labeled_size+np.delete(np.arange(X_test.shape[0]),reliable_negatives)))
        this_csr = neighbors_csr[new_indices, :][:, new_indices]

        hnc2 = HNC_pcut_2class_csr(**args_HNC)
        hnc2.fit(X_train2, y_train2)

        hnc2.fit_neighbor(this_csr)
        y_pred2 = hnc2.predict(X_test2) 

        y_actual_test = np.concatenate((y_test2, y_test[reliable_negatives]))
        y_actual_pred = np.concatenate((y_pred2, np.zeros((len(reliable_negatives), y_pred2.shape[1]))))

        for i in range(y_actual_pred.shape[1]):
            for j, metric in enumerate(metrics):
                test_scores[selector_ind, j ,y_pred.shape[1]+i] = scorers[metric](y_actual_test, y_actual_pred[:,i])
            
            positive_percentage[selector_ind,y_pred.shape[1]+i] = (y_actual_pred[:,i].sum()+labeled_size)/data_X.shape[0]

    # find the index of positive percentage closest to the true positive percentage
    closest_index = np.argmin(np.abs(positive_percentage - data_y.mean()), axis=1)
    closest_percentage = positive_percentage[np.arange(num_selectors), closest_index]
    
    return {"pos": closest_percentage[0], "acc":test_scores[0, 0, closest_index[0]], "f1":test_scores[selector_ind, 1, closest_index[0]]}

def main(arguments):

    args = process_args(arguments)
    D = readData(args.dataset)

    if args.labeledclass == 0:
        D['y'] = 1-D['y']
    
    if args.dataset == "news":
        X, y, numlabeled = news_data_prep(D, labeled_size=args.labeled, split_seed=args.seed)
    elif args.dataset == "cifar":
        X, y, numlabeled = cifar_data_prep(D, labeled_size=args.labeled, split_seed=args.seed)
    elif args.dataset == "mnist":
        X, y, numlabeled = mnist_data_prep(D, labeled_size=args.labeled, split_seed=args.seed)
    else:
        X, y, numlabeled = data_prep(D, labeled_size=args.labeled, split_seed=args.seed)
    
    # set up the parameters for HNC
    args_HNC = dict()
    args_HNC['list_lambda'] = [0.001*i for i in range(500)]
    args_HNC['epsilon'] = 0.75
    args_HNC['k'] = 5
    #args_HNC['pu_learning'] = True
    list_k = [5, 10, 15, 20, 25]

    if X.shape[0] >= 10000:
        args_HNC['list_lambda'] = [0.001*i for i in range(250)]
        args_HNC['epsilon'] = 0.25
        #args_HNC['epsilon'] = 0.5
        #args_HNC['epsilon'] = 0.75
        list_k = [5,10]
    
    print("True positive percentage: ", y.mean())


    for k in list_k:
        res = pu_hnc_csr(X, y, numlabeled, args_HNC=args_HNC, k_sparsify=k, feature_importance=args.featureweights, num_reliable_negative=numlabeled, select_neg_seed=args.negativeseed)
        print("Using k=", k)
        print("Positive percentage of chosen partition: ", res['pos'])
        print("Accuracy: ", res['acc'])
        print("F1 score: ", res['f1'])


if __name__ == '__main__':

    main(sys.argv[1:])