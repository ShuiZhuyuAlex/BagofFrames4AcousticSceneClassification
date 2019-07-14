import os
import sys
import json
import collections
import numpy as np
from Configure import data_path
from Configure import vocabulary_path

if __name__ == '__main__':
    res = collections.defaultdict(list)
    train_index_path = './train_index.json'
    if not os.path.exists(train_index_path):
        print("No Index File!!!!")
    else:
        with open(train_index_path, 'r') as trfile:
            train = json.load(trfile)
            for key in train.keys():
                print("now read fold:" + key )
                temp = []
                for uid in train[key]:
                    audio_path = os.path.join(naive_audio_path, uid)
                    for ts in os.listdir(audio_path):
                        if os.path.splitext(ts)[1] == '.mfcc':
                            print('read fold_' + key + ': ' + uid + ': ' + ts)
                            filename = os.path.join(audio_path, ts)
                            with open (filename, 'r') as sfile:
                                for phi in sfile.readlines():
                                    phi = phi.strip(',\n')
                                    phi = list(map(float, phi.split(',')))
                                    phi = np.array(phi)
                                    temp.append(phi)
                print("finish fold_" + key + ': ' + uid + ': ' + ts)
                print("write fold_" + key)
                data = np.array(temp)
                dataname = 'fold' + key + '.npy'
                data_path = os.path.join(vocabulary_path, dataname)
                np.save(data_path, data)
