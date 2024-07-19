import sys, os
from collections import Counter
from collections import defaultdict
import pandas as pd
import numpy as np
import csv

fle_pth = sys.argv[1]      # feats filename with absolute path
meta_fldr = sys.argv[2]    #  where do you want to keep these meta files
df_ctrl = pd.read_csv(fle_pth) #, sep=',', names=['audio_path', 'text', 'label'])
#get unique speaker
uni_spkrs_ctrl = df_ctrl['audio_path'].apply(lambda x: os.path.basename(x).split('_')[0]).unique()
print(uni_spkrs_ctrl)

# data prep
for idx,spkr in enumerate(uni_spkrs_ctrl):
    print(idx,uni_spkrs_ctrl[idx] )
    data_fldr = os.path.join(meta_fldr, f'set_{idx}')
    if not os.path.exists(data_fldr):
        os.makedirs(data_fldr)
    
    # test -- one speaker per set
    test_df = df_ctrl[df_ctrl['audio_path'].apply(lambda x: os.path.basename(x).split('_')[0]) == uni_spkrs_ctrl[idx]]
    test_df.to_csv(os.path.join(data_fldr,f'test_set_{idx}.csv'), sep=',', index=False)  

    # train - all speakers except the one in test
    train_df = df_ctrl[df_ctrl['audio_path'].apply(lambda x: os.path.basename(x).split('_')[0]) != uni_spkrs_ctrl[idx]]
    train_df.to_csv(os.path.join(data_fldr,f'train_set_{idx}.csv'), sep=',', index=False)

    # valid - randomly 10% from each speaker
    pos_train_df = train_df[train_df['label']==1]

    val_df.to_csv(os.path.join(data_fldr,f'val_set_{idx}.csv'), sep=',', index=False)
    print(f'Files generated for set {idx} !!!')
