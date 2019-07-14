import os
import sys
import json
import sklearn
import numpy as np
from Configure import data_path
from Configure import file_path
from Configure import model_path
from Configure import label_path
from PhiGenerator import kMeansPhiGenerator


def dataPrepare(index):
    if os.path.exits(data_path):
        X = []
        Y = []
        for uid in index:
            u_path = os.path.join(data_path, uid)
            for ts in os.listdir(path):
                file_path = os.path.join(u_path, ts)
                with open(file_path, 'r') as mfcc:
                    for frame in mfcc.readlines():
                        frame = frame.strip(',\n')
                        frame = list(map(float, frame.split(',')))
                        frame = np.array(frame)
                        temp.append(frame)
                    data = np.array(temp)
                    phi = kMeansPhiGenerator(data)
                    X.append(phi)
                """
                Get Label
                """
        return X, Y

    else:
        print("No Data")

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
    preprocess = str(args[1])
    clusters = str(args[2])
    cla = str(args[3])
    tr_index = json.load('train_index.json')
    index = tr_index[fold]
    X, Y = dataPrepare(index, preprocess, clusters)
    model(X, Y, cla)
