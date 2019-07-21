import os
import sys
import pickle
import numpy as np
from Configure import file_path
from Configure import vocabulary_path
from sklearn import preprocessing

def Norm(fold, X, method):
    normalizer_name = 'fold_' + str(fold) + '_' + method + '.p'
    normalizer_path = os.path.join(file_path, normalizer_name)
    if os.path.exists(normalizer_path):
        with open(normalizer_path,"rb") as infile:
        	normalizer = pickle.load(infile)
        phi = normalizer.transform(X)
    else:
        print("Normalizer Missing!!!")
        creatNormalizer(fold, method)
        phi = Norm(fold, X, method)
    return phi

def creatNormalizer(fold, method):
    print("Create Normalizer for fold: {f} with method: {m}".format(f=str(fold),
            m = str(method)))
    data_name = 'fold' + str(fold) + '.npy'
    data_path = os.path.join(vocabulary_path, data_name)
    if os.path.exists(data_path):
        phi = np.load(data_path)
        print("Successfully Load Data for fold: {f}".format(f=str(fold)))
        if method == 'minmaxtest':
            normalizer = preprocessing.MinMaxScaler()
        elif method == 'scale':
            normalizer = preprocessing.StandardScaler()
        else:
            normalizer = preprocessing.Normalizer()
        print("Start Fit with method: {m}".format(m=str(method)))
        normalizer.fit(phi)
        normalizer_name = 'fold_' + str(fold) + '_' + method + '.p'
        normalizer_path = os.path.join(file_path, normalizer_name)
        with open(normalizer_path, 'wb') as outfile:
            pickle.dump(normalizer, outfile)
        print("Normalizer Saved")
    else:
        print("No Feature File")


if __name__ == '__main__':
    folds = ['0', '1', '2', '3', '4']
    method = ['minmaxtest', 'scale', 'norm']
    for f in folds:
        for m in method:
            creatNormalizer(f, m)
