import pandas as pd 
import os, sys

src_file = sys.argv[1]  ## dysar csv file
dest = sys.argv[2]     ## Destination folder

# ctrl_csv = sys.argv[1]  ## ctrl csv file
# dysar_csv = sys.argv[2]  ## dysar csv file
# dest = sys.argv[3]     ## Destination folder

# df1 = pd.read_csv(ctrl_csv)
# df2 = pd.read_csv(dysar_csv)
# df = pd.concat([df1, df2], axis=0).sample(frac=1).reset_index(drop=True)

df = pd.read_csv(src_file)
uni_spkrs = df['audio_path'].apply(lambda x: os.path.basename(x).split('_')[0]).unique()
print(uni_spkrs)

##############   find common words 

wrds = []
for idx,spkr in enumerate(uni_spkrs):
    # one speaker per set
    temp_df = df[df['audio_path'].apply(lambda x: os.path.basename(x).split('_')[0]) == uni_spkrs[idx]]
    wrds_df = temp_df[temp_df['audio_path'].apply(lambda x: os.path.basename(x).split('_')[2][:2])=='UW']
    # print(spkr, wrds_df.shape[0])
    wrds.append(set(wrds_df['text']))

# print(wrds)
common_set = list(set.intersection(*map(set, wrds)))
print(len(common_set))
# sys.exit()
# cpy_wrds = common_set[:100]

uncmn_len = [50, 100, 150, 200, -1]
##########################################
for idx,num in enumerate(uncmn_len):
    cpy_wrds = common_set[:num]
    print(len(cpy_wrds))
    ### entire unknown words into test set
    data_fldr = os.path.join(dest, f'set{idx}')
    if not os.path.exists(data_fldr):
        os.makedirs(data_fldr)

    test_df = df[df['text'].isin(cpy_wrds)]  # only 100 uncmn words in test
    print(test_df.shape)
    test_new = pd.DataFrame()
    for i in uni_spkrs:
        spkr_df =test_df[test_df['audio_path'].apply(lambda x: os.path.basename(x).split('_')[0])==i]
        df_new = spkr_df.drop_duplicates(subset=['text'])
        # df_100 = df_100.drop_duplicates(subset=['text'])
        # print(df_100.shape)
        test_new = pd.concat([test_new, df_new], ignore_index=True)
    print(test_new.shape)

    test_new.sample(frac=1).reset_index(drop=True).to_csv(os.path.join(data_fldr,f'test_set{idx}.csv'), sep=',', index=False)

    # ## all common words into train
    # traincmn_df = df[df['audio_path'].apply(lambda x: os.path.basename(x).split('_')[2][:2])!='UW']
    train_df = df[~df['text'].isin(cpy_wrds)]
    print(df.shape)
    print(train_df.shape)
    train_df.sample(frac=1).reset_index(drop=True).to_csv(os.path.join(data_fldr,f'train_set{idx}.csv'), sep=',', index=False)

    # # ## 10% spkrwise from train into val
    val_df = train_df.groupby(train_df['audio_path'].apply(lambda x: os.path.basename(x).split('_')[0]), group_keys=False).apply(lambda x: x.sample(frac=0.1, random_state=42))
    val_df.sample(frac=1).reset_index(drop=True).to_csv(os.path.join(data_fldr,f'val_set{idx}.csv'), sep=',', index=False)  
print("Completed  !!!!!")