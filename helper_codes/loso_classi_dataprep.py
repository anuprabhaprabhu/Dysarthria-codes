import sys, os
from collections import Counter
from collections import defaultdict
import pandas as pd
import numpy as np
import csv

src_file = sys.argv[1]  ## dysar csv file
dest = sys.argv[2]     ## Destination folder


df_dysar = pd.read_csv(src_file)
severity_dict = { 'verylow': ['F03','M04','M12'],  # spkrs arranged
                    'low' : ['M07','F02','M16'],
                    'mid' : ['M05','M11','F04'],
                   'high' : ['M09','M10','M14']}


for idx in range(3):
    spkr = [elem[idx] for elem in severity_dict.values()]
    print(spkr)

    data_fldr = os.path.join(dest, f'set{idx}')
    if not os.path.exists(data_fldr):
        os.makedirs(data_fldr)

    # test -- one speaker per set
    test = df_dysar[df_dysar['audio_path'].apply(lambda x: os.path.basename(x).split('_')[0]).isin(spkr)]
    test.to_csv(os.path.join(data_fldr,f'test_set{idx}.csv'), sep=',', index=False)  

    # train - all speakers except the one in test
    train = df_dysar[~df_dysar['audio_path'].apply(lambda x: os.path.basename(x).split('_')[0]).isin(spkr)]
    train.to_csv(os.path.join(data_fldr,f'train_set{idx}.csv'), sep=',', index=False)

    # valid - randomly 10% from each speaker
    val = train.groupby(train['audio_path'].apply(lambda x: os.path.basename(x).split('_')[0]), group_keys=False).apply(lambda x: x.sample(frac=0.1, random_state=42))
    val.to_csv(os.path.join(data_fldr,f'val_set{idx}.csv'), sep=',', index=False)
    print(f'Files generated for set {idx} !!!')
    # sys.exit()
print('ompletted !!!!!')