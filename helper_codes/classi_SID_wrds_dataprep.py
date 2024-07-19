import pandas as pd 
import os, sys
import itertools

src_file = sys.argv[1]  ## dysar csv file
dest = sys.argv[2]     ## Destination folder

df = pd.read_csv(src_file)
severity_dict = { 'verylow': ['F03','M04','M12'],  # spkrs arranged
                    'low' : ['M07','F02','M16'],
                    'mid' : ['M05','M11','F04'],
                   'high' : ['M09','M10','M14']}

# Extracting severity levels and corresponding lists of values
severity_levels = list(severity_dict.keys())
severity_spkrs = [severity_dict[level] for level in severity_levels]
combinations = list(itertools.product(*severity_spkrs))

for idx,spkr in enumerate(combinations):
    # print(idx, spkr)
    data_fldr = os.path.join(dest, f'set{idx}')
    if not os.path.exists(data_fldr):
        os.makedirs(data_fldr)

    # ## all common words into train expect one speaker
    train_df = df[df['audio_path'].apply(lambda x: os.path.basename(x).split('_')[2][:2])!='UW']
    train_df = train_df[~train_df['audio_path'].apply(lambda x: os.path.basename(x).split('_')[0]).isin(spkr)]
    train_df.sample(frac=1).reset_index(drop=True).to_csv(os.path.join(data_fldr,f'train_set{idx}.csv'), sep=',', index=False)

    # ## 10% spkrwise from train into val
    val_df = train_df.groupby(train_df['audio_path'].apply(lambda x: os.path.basename(x).split('_')[0]), group_keys=False).apply(lambda x: x.sample(frac=0.1, random_state=42))
    val_df.sample(frac=1).reset_index(drop=True).to_csv(os.path.join(data_fldr,f'val_set{idx}.csv'), sep=',', index=False)

    ### only one speaker - uncommon words
    test_df = df[df['audio_path'].apply(lambda x: os.path.basename(x).split('_')[2][:2])=='UW']
    test_df = test_df[test_df['audio_path'].apply(lambda x: os.path.basename(x).split('_')[0]).isin(spkr)]
    test_df.sample(frac=1).reset_index(drop=True).to_csv(os.path.join(data_fldr,f'test_set{idx}.csv'), sep=',', index=False)

print("Completed  !!!!!")