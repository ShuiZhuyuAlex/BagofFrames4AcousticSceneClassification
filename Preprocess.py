import os
import sys
import pickle
import numpy as np
from Configure import root_path
from Configure import data_path
from sklearn import preprocessing

def Norm(fold, X, method):
    normalizer_name = 'fold_' + str(fold) + '_' + method + '.p'
    normalizer_path = os.path.join(root_path, normalizer_name)
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
    data_path = os.path.join(vocabulary_path, data_name)
    if os.path.exists(data_path):
        phi = np.load(data_path)
        if method == 'minmaxtest':
            normalizer = preprocessing.MinMaxScaler()
        elif method == 'scale':
            normalizer = preprocessing.StandardScaler()
        else:
            normalizer = preprocessing.Normalizer()
        normalizer.fit(phi)
        normalizer_name = 'fold_' + str(fold) + '_' + method + '.p'
        normalizer_path = os.path.join(root_path, normalizer_name)
        with open(normalizer_path, 'wb') as outfile:
            pickle.dump(normalizer, outfile)
        print("Normalizer Saved")
    else:
        print("No Feature File")


if __name__ == '__main__':
    args = sys.argv[1].split(',')
    print(args)
    fold = args[0]
    method = args[1]
    creatNormalizer(fold, method)
