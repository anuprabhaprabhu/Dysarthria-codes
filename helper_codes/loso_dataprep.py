import sys, os
from collections import Counter
from collections import defaultdict
import pandas as pd
import csv

pth = sys.argv[1]

df_ctrl = pd.read_csv(os.path.join(pth, 'ctrl_only2sec_M5.csv')) #, sep=',', names=['audio_path', 'text', 'label'])
df_dysar = pd.read_csv(os.path.join(pth, 'dysar_only2sec_M5.csv'))
#get unique speaker
uni_spkrs_ctrl = df_ctrl['audio_path'].apply(lambda x: os.path.basename(x).split('_')[0]).unique()
print(uni_spkrs_ctrl)
print(type(uni_spkrs_ctrl))
uni_spkrs_dysar = df_dysar['audio_path'].apply(lambda x: os.path.basename(x).split('_')[0]).unique()
print(uni_spkrs_dysar)

#### need to take care of the mismatch between the control & Dysarthric speakers
# data prep
for idx,spkr in enumerate(uni_spkrs_ctrl):
    print(idx,uni_spkrs_ctrl[idx] )
    print(idx,uni_spkrs_dysar[idx] )

    data_fldr = os.path.join(pth, f'set_{idx}')
    if not os.path.exists(data_fldr):
        os.makedirs(data_fldr)
    
    # test -- one speaker per set
    test1 = df_ctrl[df_ctrl['audio_path'].apply(lambda x: os.path.basename(x).split('_')[0]) == uni_spkrs_ctrl[idx]]
    test2 = df_dysar[df_dysar['audio_path'].apply(lambda x: os.path.basename(x).split('_')[0]) == uni_spkrs_dysar[idx]]
    test_df = pd.concat([test1, test2], axis=0).sample(frac=1).reset_index(drop=True)
    test_df.to_csv(os.path.join(data_fldr,f'test_set_{idx}.csv'), sep=',', index=False)  

    # train - all speakers except the one in test
    train1 = df_ctrl[df_ctrl['audio_path'].apply(lambda x: os.path.basename(x).split('_')[0]) != uni_spkrs_ctrl[idx]]
    train2 = df_dysar[df_dysar['audio_path'].apply(lambda x: os.path.basename(x).split('_')[0]) != uni_spkrs_dysar[idx]]
    train_df = pd.concat([train1, train2], axis=0).sample(frac=1).reset_index(drop=True)
    train_df.to_csv(os.path.join(data_fldr,f'train_set_{idx}.csv'), sep=',', index=False)

    # valid - randomly 10% from each speaker
    val1 = train1.groupby(train1['audio_path'].apply(lambda x: os.path.basename(x).split('_')[0]), group_keys=False).apply(lambda x: x.sample(frac=0.1, random_state=42))
    val2 = train2.groupby(train2['audio_path'].apply(lambda x: os.path.basename(x).split('_')[0]), group_keys=False).apply(lambda x: x.sample(frac=0.1, random_state=42))
    val_df = pd.concat([val1, val2], axis=0).sample(frac=1).reset_index(drop=True)
    val_df.to_csv(os.path.join(data_fldr,f'val_set_{idx}.csv'), sep=',', index=False)
    print(f'Files generated for set {idx} !!!')
    sys.exit()