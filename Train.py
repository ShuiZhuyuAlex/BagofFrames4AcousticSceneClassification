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
        filename = user + '.feature_labels.csv'
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
            raw_data = np.array(temp
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
    args = sys.argv[1]
    fold = str(args[0])
    method = str(args[1])
    clusters = str(args[2])
    cla = str(args[3])
    tr_index = json.load('train_index.json')
    index = tr_index[fold]
    X, Y = dataPrepare(fold, index, method, clusters)
    model(X, Y, cla)
