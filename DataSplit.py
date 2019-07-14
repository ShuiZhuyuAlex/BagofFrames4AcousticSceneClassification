import os
import json
import h5py as h5
import collections

folds_path = '/Users/shuizhuyu/Desktop/cv_5_folds'
feature_path = './'

train = collections.defaultdict(list)
test = collections.defaultdict(list)
folds = os.listdir(folds_path)


for foldname in folds:
    filename = os.path.join(folds_path, foldname)
    with open(filename, 'r') as idlist:
        uids = [uuid.strip('\n') for uuid in idlist]
        if 'train' in filename:
            if 'fold_0' in filename:
                train['0'].extend(uids)
            elif 'fold_1' in filename:
                train['1'].extend(uids)
            elif 'fold_2' in filename:
                train['2'].extend(uids)
            elif 'fold_3' in filename:
                train['3'].extend(uids)
            else:
                train['4'].extend(uids)
        else:
            if 'fold_0' in filename:
                test['0'].extend(uids)
            elif 'fold_1' in filename:
                test['1'].extend(uids)
            elif 'fold_2' in filename:
                test['2'].extend(uids)
            elif 'fold_3' in filename:
                test['3'].extend(uids)
            else:
                test['4'].extend(uids)

train_index_path = './train_index.json'
test_index_path = './test_index.json'

tr_json = json.dumps(train)
with open(train_index_path, 'w') as tr:
    tr.write(tr_json)

ts_json = json.dumps(train)
with open(test_index_path, 'w') as ts:
    ts.write(ts_json)
