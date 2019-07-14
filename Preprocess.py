import os
import pickle
import numpy as np
from sklearn import preprocessing


def Norm(fold, X, method):
    normalizer_name = 'fold_' + str(fold) + '_' + method + '.p'
    normalizer_path = os.path.join(path, normalizer_name)
    if os.path.exists(normalizer_path):
        with open(normalizer_path,"rb") as infile:
        	normalizer = pickle.load(infile)
        phi = normalizer.transform(X)
        return phi
    else:
        print("Normalizer Missing!!!")
        creatNormalizer(fold, method)
        Norm(fold, X, method)

def creatNormalizer(fold, method):
    print("Create Normalizer")
    data_name = 'fold_' + str(fold) + '.npy'
    data_path = os.path.join(path, data_name)
    if os.path.exists(data_path):
        phi = np.load(data_path)
        if method == 'minxtest':
            normalizer = preprocessing.MinMaxScaler()
        elif method == 'scale':
            normalizer = preprocessing.StandardScaler()
        else:
            normalizer = preprocessing.Normalizer()
        normalizer.fit(phi)
        normalizer_name = 'fold_' + str(fold) + '_' + method + '.p'
        normalizer_path = os.path.join(path, normalizer_name)
        with open(normalizer_path, 'wb') as outfile:
            pickle.dump(normalizer, outfile)
        print("Normalizer Saved")
    else:
        print("No Feature File")
