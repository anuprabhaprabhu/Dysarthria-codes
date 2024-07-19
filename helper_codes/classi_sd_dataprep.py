import pandas as pd 
import os, sys

src_file = sys.argv[1]  ## dysar csv file
dest = sys.argv[2]     ## Destination folder

df = pd.read_csv(src_file)
uni_spkrs = df['audio_path'].apply(lambda x: os.path.basename(x).split('_')[0]).unique()
print(uni_spkrs)

# ## all common words into train
train_df = df[df['audio_path'].apply(lambda x: os.path.basename(x).split('_')[2][:2])!='UW']

# ## 10% spkrwise from train into val
val_df = train_df.groupby(train_df['audio_path'].apply(lambda x: os.path.basename(x).split('_')[0]), group_keys=False).apply(lambda x: x.sample(frac=0.1, random_state=42))

## 5 different sets from uncommon words into test
UK_wrds_df = df[df['audio_path'].apply(lambda x: os.path.basename(x).split('_')[2][:2])=='UW']
UK_wrds = UK_wrds_df['text'].unique()
print(f'Number unique uncommon words in the given csv file: {len(UK_wrds)}')

wrds = []
for idx,spkr in enumerate(uni_spkrs):
    # one speaker per set
    temp_df = UK_wrds_df[UK_wrds_df['audio_path'].apply(lambda x: os.path.basename(x).split('_')[0]) == uni_spkrs[idx]]
    wrds.append(set(temp_df['text']))
UK_wrds = list(set.intersection(*map(set, wrds)))
print(f'Number unique uncommon words covered by all speakers in the given csv file: {len(UK_wrds)}')

set = [50,100,150,200,250]
for idx,i in enumerate(set):
    wrds_lst = UK_wrds[:i]
    test_df = UK_wrds_df[UK_wrds_df['text'].isin(wrds_lst)]

    data_fldr = os.path.join(dest, f'set{idx}')
    if not os.path.exists(data_fldr):
        os.makedirs(data_fldr)

    test_df.to_csv(os.path.join(data_fldr,f'test_set{idx}.csv'), sep=',', index=False)
    train_df.to_csv(os.path.join(data_fldr,f'train_set{idx}.csv'), sep=',', index=False)
    val_df.to_csv(os.path.join(data_fldr,f'val_set{idx}.csv'), sep=',', index=False)
idx=4    
### entire unknown words into test set
data_fldr = os.path.join(dest, f'set{idx+1}')
if not os.path.exists(data_fldr):
    os.makedirs(data_fldr)

UK_wrds_df.to_csv(os.path.join(data_fldr,f'test_set{idx+1}.csv'), sep=',', index=False)
train_df.to_csv(os.path.join(data_fldr,f'train_set{idx+1}.csv'), sep=',', index=False)
val_df.to_csv(os.path.join(data_fldr,f'val_set{idx+1}.csv'), sep=',', index=False)  

print("Completed  !!!!!")