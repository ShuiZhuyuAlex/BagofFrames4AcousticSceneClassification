import pandas as pd
import numpy as np
import os

filepath = '/Users/shuizhuyu/Desktop/Labels'
outpath = '/Users/shuizhuyu/Desktop/SummerResearch/Labels'

label_dict = dict()
for file in os.listdir(filepath):
    if os.path.splitext(file)[-1] != '.csv':
        continue
    csv_path = os.path.join(filepath, file)
    csv_file = pd.read_csv(csv_path, header=0)
    try:
        df = csv_file[['timestamp','label:LYING_DOWN','label:BICYCLING','label:SLEEPING','label:IN_CLASS','label:IN_A_MEETING','label:IN_A_CAR','label:ON_A_BUS','label:DRIVE_-_I_M_THE_DRIVER','label:FIX_restaurant','label:DRINKING__ALCOHOL_']]
    except Exception as e:
        print(file)
    df = df.fillna(float(-1), inplace=False)
    h5_path = os.path.join(outpath, file) + '.h5'
    h5 = pd.HDFStore(h5_path, 'w')
    for line in df.values:
        time = str(int(line[0]))
        h5[time] = np.random.random(100)
    h5.close()