import os
import sys
import pickle
import numpy as np
from Configure import root_path
from Preprocess import Norm
from sklearn.cluster import MiniBatchKMeans


def kMeansTrain(fold, method, clusters):
    print("Create KMeans Model")
    data_name = 'fold_' + str(fold) + '.npy'
    data_path = os.path.join(vocabulary_path, data_name)
    if os.path.exists(data_path):
        temp = np.load(data_path)
        train_data = Norm(fold, temp, method)
        kmeans = MiniBatchKMeans(n_clusters = clusters, max_iter = 300,
                                 n_init = 30, batch_size = 200,
                                 init = 'k-means++',
                                 reassignment_ratio = 0.007).fit(train_data)
        model_name = ('fold_' + str(fold) +  '_' +
                      method + '_' + str(clusters) + '.p')
        model_path = os.path.join(root_path, model_name)
        with open(model_path, "wb") as outfile:
        	pickle.dump(kmeans,outfile)
        print("KMeans generated ans saved")
    else:
        print("No Feature File")


def kMeansPhiGenerator(fold, X, method, clusters):
    model_name = 'fold_' + str(fold) +  '_' + method + '_' + str(clusters) + '.p'
    model_path = os.path.join(root_path, model_name)
    if os.path.exists(model_path):
        with open(model_path, "rb") as infile:
        	kmeans = pickle.load(infile)
        no_of_centers = kmeans.cluster_centers_.shape[0]
        X = Norm(fold, X, method)
        kmeans_output = kmeans.predict(X)
        unique, counts = np.unique(kmeans_output, return_counts=True)
        phi = [0]*no_of_centers
        for j in range(no_of_centers):
            if j in unique:
                phi[j] = counts[np.where(unique == j)][0]
    else:
        kMeansTrain(fold, method, clusters)
        kMeansPhiGenerator(fold, X, method, clusters)

if __name__ == '__main__':
    args = sys.argv[1].split(',')
    fold = args[0]
    method = args[1]
    clusters = args[2]
    kMeansTrain(fold, method, clusters)
