import os
import sys
import json
import sklearn
import numpy as np
import pandas as pd
from Configure import data_path
from Configure import file_path
from Configure import model_path
from Configure import label_path
from PhiGenerator import kMeansPhiGenerator


def dataPrepare(index, fold, method, clusters):
    X = list()
    Y = list()
    for user in index:
        filename = user + '.features_labels.csv'
        lpath = os.path.join(label_path, filename)
        temppath= os.path.join(data_path, user)
        df = pd.read_csv(lpath, header=0, index_col=0)
        for ts in df.index():
            y = numpy.array(list(map(float, ds.loc[ts])))
            sname = ts + '.sound.mfcc'
            dpath = os.path.join(temppath, sanme)
            temp = list()
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
    Y = np.array(Y)
    X = np.array(X)
    return X, Y

def model(X, Y, cla):
    if cla == 'SVM':
        pass
    elif cla == 'Adaboost':
        pass
    else:
        pass


if __name__ == '__main__':
    with open('train_index.json', 'r') as index_file:
        all = json.load(index_file)
    index = all['0']
    x, y = dataPrepare(index, '0', 'scale', 70)
    print(x.shape)
    print(y.shape)
