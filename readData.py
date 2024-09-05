import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_svmlight_file, fetch_openml
from sklearn.model_selection import train_test_split
import pandas as pd
from keras.datasets import cifar10
import ssl
from scipy.io.arff import loadarff 

ssl._create_default_https_context = ssl._create_unverified_context

def readData(data_name, path = 'data/'):

    if data_name == "vote":
        ## vote data
        data =  pd.read_csv(path+'congressional+voting+records/house-votes-84.data', sep=",", header=None,index_col=False)
        data_X = data.iloc[:,1:].map(lambda x: 1 if x=='y' else 0 if 'x' == '?' else -1).values
        data_y = data.iloc[:,:1].apply(lambda x: 0 if x[0] == 'republican' else 1, axis=1).values
        return {'X': data_X, 'y': data_y}
    
    elif data_name == "obesity":
        ## obesity data
        data =  pd.read_csv(path+'obesity.csv', sep=",", header=0,index_col=False)
        labels_caec = [0 for i in range(len(data))]
        labels_calc = [0 for i in range(len(data))]
        labels = [0 for i in range(len(data))]

        for i in range(len(data)):
            this_caec = data['CAEC'][i]
            if this_caec == 'no':
                labels_caec[i] = 0
            elif this_caec == 'Sometimes':
                labels_caec[i] = 1
            elif this_caec == 'Frequently':
                labels_caec[i] = 2
            elif this_caec == 'Always':
                labels_caec[i] = 3

        for i in range(len(data)):
            this_calc = data['CALC'][i]
            if this_calc == 'no':
                labels_calc[i] = 0
            elif this_calc == 'Sometimes':
                labels_calc[i] = 1
            elif this_calc == 'Frequently':
                labels_calc[i] = 2
            elif this_calc == 'Always':
                labels_calc[i] = 3

        for i in range(len(data)):
            if data['NObeyesdad'][i][0:7] == 'Obesity':
                labels[i] = 1

        data = data.drop(columns=['CAEC', 'CALC', 'NObeyesdad']) 
        data = pd.get_dummies(data, columns=['Gender', 'family_history_with_overweight', 'FAVC', 'SMOKE', 'SCC', 'MTRANS'], drop_first=True)

        data['caec_num'] = labels_caec
        data['calc_num'] = labels_calc
        data['labels'] = labels
        data_X = data.drop(columns=['labels'])
        data_y = data[['labels']]
        return {'X': data_X, 'y': data_y.values.ravel()}
    
    elif data_name == "mushroom":
        ## mushroom data
        data = load_svmlight_file(path+'mushrooms.txt')
        data_X = np.array(data[0].todense())
        data_y = np.array(data[1])-1
        return {'X': data_X, 'y': data_y}
    
    elif data_name == "news":
        ## news data
        return {'X': np.load(path+"embedded_news_data_300dim_X.npy"), 'y': np.load(path+"embedded_news_data_300dim_X.npy")}
    
    elif data_name == "letter":
        ## letter data
        data =  pd.read_csv(path+'letter-recognition.data', sep=",", header=None,index_col=False)
        labels = [0 for i in range(len(data))]
        for i in range(len(data)):
            if ord(data[0][i]) <= 77:
                labels[i] = 1
        data['labels'] = labels
        data = data.drop(columns=[data.columns[0]])
        data_X = data[data.columns[:-1]]
        data_y = data[['labels']]
        data_y = data_y.values.ravel()
        return {'X': data_X, 'y': data_y}
    
    elif data_name == "cifar":
        ## cifar data
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        data_X = np.concatenate((x_train/255, x_test/255),axis=0).reshape((60000,3072))
        data_y = np.concatenate((y_train, y_test),axis=0).flatten()
        return {'X': data_X, 'y': data_y}
    
    elif data_name == "mnist":
        ## mnist data
        mnist = fetch_openml('mnist_784', return_X_y=True)
        data_X = mnist[0].values
        data_y = np.array(mnist[1].values).astype(int)
        return {'X': data_X, 'y': data_y}
    
    elif data_name == "wine":
        ## wine
        data = pd.read_csv(path+'winequality-red.csv', header=0, index_col=False)
        data_X = data[data.columns[:-1]]
        labels = [0 for i in range(len(data))]
        for i in range(len(data)):
            if data[data.columns[-1]][i] > 5:
                labels[i] = 1
        data['labels'] = labels
        data_y = data[['labels']]
        data_y = data_y.values.ravel()
        return {'X': data_X, 'y': data_y}
    
    elif data_name == "blood":
        data = pd.read_csv(path+"transfusion.data")
        data_X = data.iloc[:,:-1].values
        data_y = data.iloc[:,-1].values
        return {'X': data_X, 'y': data_y}
    
    elif data_name == "maternal":
        data =  pd.read_csv(path+'maternalhealth.csv', sep=",", header=0,index_col=False)
        labels = [1 for i in range(len(data))]
        for i in range(len(data)):
            if data['RiskLevel'][i] == 'low risk':
                labels[i] = 0
        data = data.drop(columns=['RiskLevel']) 
        data['labels'] = labels
        data_X = data.drop(columns=['labels']) 
        return {'X': data_X, 'y': data[['labels']].values.ravel()}
    
    elif data_name == "cardio":
        data =  pd.read_excel(path+'CTG.xls', sheet_name="Data", header = 1)
        data_X = data[data.columns[10:31]][:-3]
        labels = [1 if data['NSP'][i]==1 else 0 for i in range(len(data_X))]
        return {'X': data_X, 'y': np.array(labels)}
    
    elif data_name == "german":
        data =  pd.read_csv(path+'german.data-numeric', delim_whitespace=True,header=None)
        data_X = data[data.columns[:-1]]
        data_y = data[data.columns[-1]]
        for i in range(len(data_y)):
            data_y[i] -= 1
        data_y = data_y.values.ravel()
        return {'X':data_X, 'y':data_y}
    
    elif data_name == "breast":
        data = pd.read_csv(path+'wdbc.data', sep=",", header=None, index_col=0)
        data_X = data.iloc[:,1:].values
        data_y = (data[1]=='M').astype(int).values # malignant = 1, benign  = 0
        return {'X': data_X, 'y': data_y}
    
    elif data_name == "spambase":
        data = pd.read_csv(path+'spambase.data', sep=",", header=None, index_col=False)
        data_X = data.iloc[:,:-1].values
        data_y = data.iloc[:,-1].values
        return {'X': data_X, 'y': data_y}
    
    elif data_name == "credit":
        data = pd.read_excel(path+'credit-default.xls', header=1, index_col=0)
        data_X = data.iloc[:,:-1].values
        data_y = data.iloc[:,-1].values
        return {'X': data_X, 'y': data_y} 

    elif data_name == "taiwan":
        data = pd.read_csv(path+'taiwanese_bank_data.csv', sep=",", header=0, index_col=False)  
        data_X = data.iloc[:,1:].values
        data_y = data.iloc[:,0].values
        return {'X': data_X, 'y': data_y} 
    
    elif data_name == "phishing":
        data = pd.DataFrame(loadarff(path+'phishing+websites/Training Dataset.arff')[0])
        data = data.select_dtypes([object])
        data = data.stack().str.decode('utf-8').unstack()
        data_X = data.iloc[:,:-1].astype(float).values
        data_y = ((data.iloc[:,-1].astype(int)+1)//2).values
        return {'X': data_X, 'y': data_y}
    
    elif data_name == "pageblock":
        data = pd.DataFrame(loadarff(path+'dataset_30_page-blocks.arff')[0])
        data_X = data.drop(columns=['class']).astype(float).values
        data_y = data[['class']].stack().str.decode('utf-8').unstack().astype(int)
        data_y = data_y.apply(lambda x: 1 if x.iloc[0] == 1 else 0, axis=1).values
        return {'X': data_X, 'y': data_y}


def data_prep(data, labeled_class=1, labeled_size=500, split_seed=2539):

    labeled_class_indices = np.where(data['y'] == labeled_class)[0]
    unlabeled_class_indices = np.where(data['y'] == 1-labeled_class)[0]

    labeled_unlabeled_indices, labeled_indices = train_test_split(labeled_class_indices, test_size=labeled_size, random_state=split_seed)

    X_all = StandardScaler().fit_transform(data['X'])
    y_all = data['y']

    X = np.concatenate((X_all[labeled_indices,:], X_all[labeled_unlabeled_indices,:], X_all[unlabeled_class_indices,:]))
    y = np.concatenate((y_all[labeled_indices], y_all[labeled_unlabeled_indices], y_all[unlabeled_class_indices]))
    
    return X, y, len(labeled_indices)

def news_data_prep(data, labeled_class=1, labeled_size=500, split_seed=2539):

    if labeled_class == 1:
        positive_classes = [0,1,2,3,4,5,6,7,8,9,10]
    else:
        positive_classes = [11,12,13,14,15,16,17,18,19]
    positive_mask = np.isin(data['y'].flatten(), positive_classes)

    np.random.seed(split_seed)
    labeled_indices = np.concatenate(tuple([np.random.choice(np.where(data['y'] == i)[0], labeled_size//len(positive_classes), replace=False) for i in positive_classes]))
    labeled_class_indices = np.where(positive_mask)[0]
    labeled_unlabeled_indices = np.array([i for i in labeled_class_indices if i not in labeled_indices])
    unlabeled_class_indices = np.where(np.invert(positive_mask))[0]

    X_all = StandardScaler().fit_transform(data['X'])

    y_all = positive_mask.astype(int)

    X = np.concatenate((X_all[labeled_indices,:], X_all[labeled_unlabeled_indices,:], X_all[unlabeled_class_indices,:]))
    y = np.concatenate((y_all[labeled_indices], y_all[labeled_unlabeled_indices], y_all[unlabeled_class_indices]))

    return X, y, len(labeled_indices)

def mnist_data_prep(data, labeled_digits=[1,3,5,7,9], unlabeled_digits=None, labeled_size=500, standardize=True, split_seed=2539):

    # labeled_size: number of labeled samples per digit in labeled_digits

    if unlabeled_digits is None:
        unlabeled_digits = [i for i in range(10) if i not in labeled_digits]

    np.random.seed(split_seed)
    labeled_indices = np.concatenate(tuple([np.random.choice(np.where(data['y'] == i)[0], labeled_size, replace=False) for i in labeled_digits]))
    labeled_class_indices = np.concatenate(tuple([np.where(data['y'] == i)[0] for i in labeled_digits]))
    labeled_unlabeled_indices = np.array([i for i in labeled_class_indices if i not in labeled_indices])
    unlabeled_class_indices = np.concatenate(tuple([np.where(data['y'] == i)[0] for i in unlabeled_digits]))

    y_all = np.array([1 if data['y'][i] in labeled_digits else 0 for i in range(data['X'].shape[0])])

    X = np.concatenate((data['X'][labeled_indices,:], data['X'][labeled_unlabeled_indices,:], data['X'][unlabeled_class_indices,:]))
    y = np.concatenate((y_all[labeled_indices], y_all[labeled_unlabeled_indices], y_all[unlabeled_class_indices]))

    if standardize:
        X = StandardScaler().fit_transform(X)

    return X, y, len(labeled_indices)

def cifar_data_prep(data, labeled_class=1, labeled_size=500, split_seed=2539):

    if labeled_class == 1:
        positive_classes = [2,3,4,5,6,7]
    else:
        positive_classes = [0,1,8,9]
    positive_mask = np.isin(data['y'].flatten(), positive_classes)

    np.random.seed(split_seed)
    labeled_indices = np.concatenate(tuple([np.random.choice(np.where(data['y'] == i)[0], labeled_size//len(positive_classes), replace=False) for i in positive_classes]))
    labeled_class_indices = np.where(positive_mask)[0]
    labeled_unlabeled_indices = np.array([i for i in labeled_class_indices if i not in labeled_indices])
    unlabeled_class_indices = np.where(np.invert(positive_mask))[0]

    X_all = StandardScaler().fit_transform(data['X'])

    y_all = positive_mask.astype(int)

    X = np.concatenate((X_all[labeled_indices,:], X_all[labeled_unlabeled_indices,:], X_all[unlabeled_class_indices,:]))
    y = np.concatenate((y_all[labeled_indices], y_all[labeled_unlabeled_indices], y_all[unlabeled_class_indices]))

    return X, y, len(labeled_indices)