import sys, os
from collections import Counter
from collections import defaultdict
import pandas as pd
import numpy as np
import itertools

fle_pth = sys.argv[1]      # feats filename with absolute path
meta_fldr = sys.argv[2]    #  where do you want to keep these meta files
# ls_train100 = sys.argv[3]
df_ctrl = pd.read_csv(fle_pth) #, sep=',', names=['audio_path', 'text', 'label'])
# ls_100 = pd.read_csv(ls_train100)
#get unique speaker
uni_spkrs_ctrl = df_ctrl['audio_path'].apply(lambda x: os.path.basename(x).split('_')[0]).unique()
print(uni_spkrs_ctrl)
# all_combinations = list(itertools.combinations(uni_spkrs_ctrl, 7))
# filtered_combinations = [combo for combo in all_combinations if 1 <= sum(1 for var in combo if var.startswith('CF')) < 2]
# # print(len(filtered_combinations))
# cnt=0
# # Print the filtered combinations
# for combo in filtered_combinations:
#     cnt+=1
#     # print(combo)
# print(cnt)

test_spkrs = ['CF03','CF04','CM04','CM05','CM12','CM13']
# # test -- 4 speaker per set
# test_df = df_ctrl[df_ctrl['audio_path'].apply(lambda x: os.path.basename(x).split('_')[0]).isin(test_spkrs)]
# test_df.to_csv(os.path.join(meta_fldr,f'test_set6.csv'), sep=',', index=False)

# #### train kws type
train_df = df_ctrl[df_ctrl['audio_path'].apply(lambda x: os.path.basename(x).split('_')[0]).isin(test_spkrs)]
# train_df.to_csv(os.path.join(meta_fldr,f'train_set.csv'), sep=',', index=False)
##############################
# # train kws type + one word samples from ls train100
# train1 = df_ctrl[~df_ctrl['audio_path'].apply(lambda x: os.path.basename(x).split('_')[0]).isin(test_spkrs)]
# ls_100_df = ls_100[ls_100['text'].apply(lambda x: len(x.split()) == 1)] #  get single words

############## This is done to reduce number of unique words in LS-KWS ##########
# ls_100_df1 = ls_100_df[ls_100_df['label']==1] # get unique audio files
# uni_wrds_ls100_1wrd = set(ls_100_df1['text']) 
# # generate negative samples for KWS
# new_ls=[]
# for _,line in ls_100_df1.iterrows():
#     new_txt = np.random.choice(list(uni_wrds_ls100_1wrd - {line['text']}))
#     new_ls.append([line['audio_path'], line['text'], line['label']])
#     new_ls.append([line['audio_path'], new_txt, 0])
# new_ls_df = pd.DataFrame(new_ls, columns=['audio_path', 'text', 'label'])
#                 ################################################

# train_df = pd.concat([df_ctrl, new_ls_df], axis=0).sample(frac=1).reset_index(drop=True)
# train_df.to_csv(os.path.join(meta_fldr,f'train_set5.csv'), sep=',', index=False)

### # valid - randomly 10% from each speaker from train of UA
pos_train_df = train_df[train_df['label']==1]
temp_df = pos_train_df.groupby(pos_train_df['audio_path'].apply(lambda x: os.path.basename(x).split('_')[0]), group_keys=False).apply(lambda x: x.sample(frac=0.1, random_state=42))
# val =[]
# for _,line in temp_df.iterrows():
#     val.append(train_df.loc[train_df['audio_path']== line['audio_path']])
#     print(val)

# val_df = train_df.loc[train_df['audio_path']== pos_train_df['audio_path']]
val_df = train_df.loc[train_df['audio_path'].isin(temp_df['audio_path'])]
# print(val_df.head())
# sys.exit()
# val_df = pd.DataFrame(val, columns=['audio_path', 'text', 'label'])
val_df.to_csv(os.path.join(meta_fldr,f'val_set6.csv'), sep=',', index=False)


print('Data prep is done !!!!')