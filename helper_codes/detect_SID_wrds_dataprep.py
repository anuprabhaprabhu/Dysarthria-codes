import pandas as pd 
import numpy as np
import os, sys

ctrl_csv = sys.argv[1]  ## ctrl csv file
dysar_csv = sys.argv[2]  ## dysar csv file
dest = sys.argv[3]     ## Destination folder

df1 = pd.read_csv(ctrl_csv)
df2 = pd.read_csv(dysar_csv)
df = pd.concat([df1, df2], axis=0).sample(frac=1).reset_index(drop=True)

uni_spkrs_ctrl = df1['audio_path'].apply(lambda x: os.path.basename(x).split('_')[0]).unique()
np.random.shuffle(uni_spkrs_ctrl)
print(uni_spkrs_ctrl)
uni_spkrs_dysar = df2['audio_path'].apply(lambda x: os.path.basename(x).split('_')[0]).unique()
print(uni_spkrs_dysar)
uni_spkrs = []
for id, i in enumerate(uni_spkrs_ctrl):
    uni_spkrs.append([uni_spkrs_ctrl[id], uni_spkrs_dysar[id]])
uni_spkrs.append(['CF03', uni_spkrs_dysar[id+1]])
print(uni_spkrs)

for idx,i in enumerate(uni_spkrs):
    data_fldr = os.path.join(dest, f'set{idx}')
    if not os.path.exists(data_fldr):
        os.makedirs(data_fldr)
    print(idx, i)
    # ## all common words into train expect one speaker
    train_df = df[df['audio_path'].apply(lambda x: os.path.basename(x).split('_')[2][:2])!='UW']
    train_df = train_df[~train_df['audio_path'].apply(lambda x: os.path.basename(x).split('_')[0]).isin(i)]
    train_df.sample(frac=1).reset_index(drop=True).to_csv(os.path.join(data_fldr,f'train_set{idx}.csv'), sep=',', index=False)

    # ## 10% spkrwise from train into val
    val_df = train_df.groupby(train_df['audio_path'].apply(lambda x: os.path.basename(x).split('_')[0]), group_keys=False).apply(lambda x: x.sample(frac=0.1, random_state=42))
    val_df.sample(frac=1).reset_index(drop=True).to_csv(os.path.join(data_fldr,f'val_set{idx}.csv'), sep=',', index=False)

    ### only one speaker - uncommon words
    test_df = df[df['audio_path'].apply(lambda x: os.path.basename(x).split('_')[2][:2])=='UW']
    test_df = test_df[test_df['audio_path'].apply(lambda x: os.path.basename(x).split('_')[0]).isin(i)]
    test_df.sample(frac=1).reset_index(drop=True).to_csv(os.path.join(data_fldr,f'test_set{idx}.csv'), sep=',', index=False)

print("Completed  !!!!!")