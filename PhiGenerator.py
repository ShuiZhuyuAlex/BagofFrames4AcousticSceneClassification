import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from Preprocess import Norm
from Configure import data_path
from Configure import file_path
from Configure import label_path
from Configure import vocabulary_path
from sklearn.cluster import MiniBatchKMeans


def kMeansTrain(fold, method, clusters):
    print("Create KMeans Model for fold: {f}".format(f = fold))
    print("Number of Clusters: {c}".format(c = str(clusters)))
    print("Normalization Method: {m}".format(m = method))
    data_name = 'fold' + str(fold) + '.npy'
    x_path = os.path.join(vocabulary_path, data_name)
    if os.path.exists(x_path):
        temp = np.load(x_path)
        train_data = Norm(fold, temp, method)
        kmeans = MiniBatchKMeans(n_clusters = clusters, max_iter = 300,
                                 n_init = 30, batch_size = 200,
                                 init = 'k-means++',
                                 reassignment_ratio = 0.007).fit(train_data)
        cluster_name = ('fold_' + str(fold) +  '_' +
                      method + '_' + str(clusters) + '.p')
        cluster_path = os.path.join(file_path, cluster_name)
        with open(cluster_path, "wb") as outfile:
        	pickle.dump(kmeans,outfile)
        print("KMeans generated and saved")
    else:
        print("No Feature File")


def kMeansPhiGenerator(fold, X, method, clusters):
    cluster_name = ('fold_' + str(fold) +  '_' + method +
                    '_' + str(clusters) + '.p')
    cluster_path = os.path.join(file_path, cluster_name)
    if os.path.exists(cluster_path):
        with open(cluster_path, "rb") as infile:
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
        phi = kMeansPhiGenerator(fold, X, method, clusters)

    return phi

def dataFactory(index, fold, method, clusters):
    print("Generate Phi for fold: {f}".format(f=fold))
    X = list()
    Y = list()
    for user in index:
        print("User: {u}".format(u = user))
        filename = user + '.features_labels.csv'
        lpath = os.path.join(label_path, filename)
        temppath= os.path.join(data_path, user)
        df = pd.read_csv(lpath, header=0, index_col=0)
        for ts in df.index:
            y = np.array(list(map(float, df.loc[ts])))
            sname = str(ts) + '.sound.mfcc'
            dpath = os.path.join(temppath, sname)
            if os.path.exists(dpath):
                temp = list()
                print("------------->Time Stamp: {ts}".format(ts = sname))
                with open(dpath, 'r') as sf:
                    for frame in sf.readlines():
                        frame = frame.strip(',\n')
                        frame = list(map(float, frame.split(',')))
                        frame = np.array(frame)
                        temp.append(frame)
                raw_data = np.array(temp)
                x = kMeansPhiGenerator(fold, raw_data, method, clusters)
                Y.append(y)
                X.append(x)
            else:
                print("------------->Data Missing For: {ts}".format(ts = sname))
                continue
    Y = np.array(Y)
    X = np.array(X)
    x_name = "fold{f}_method{m}_clusters{c}_data.npy".format(f=fold, m=method, c=clusters)
    y_name = "fold{f}_method{m}_clusters{c}_label.npy".format(f=fold, m=method, c=clusters)
    x_path = os.path.join(data_path, x_name)
    y_path = os.path.join(data_path, y_name)
    np.save(x_path, X)
    np.save(y_path, Y)
    return X, Y

if __name__ == '__main__':
    with open('train_index.json', 'r') as index_file:
        all = json.load(index_file)
    index = all['0']
    x, y = dataFactory(index, '0', 'scale', 70)
    print(x.shape)
    print(y.shape)
    # folds = ['0', '1', '2', '3', '4']
    # clusters = [50, 60, 70, 80, 90, 100]
    # method = ['minmaxtest', 'scale', 'norm']
    # for f in folds:
    #     for c in clusters:
    #         for m in method:
    #             kMeansTrain(f, m, c)
