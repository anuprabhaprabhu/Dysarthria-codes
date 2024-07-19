
import sys, os
import pandas as pd 


fle = sys.argv[1]

df = pd.read_csv(fle)
uni_spkrs = df['audio_path'].apply(lambda x: os.path.basename(x).split('_')[0]).unique()
print(uni_spkrs)

wrds = []
for idx,spkr in enumerate(uni_spkrs):
    # one speaker per set
    temp_df = df[df['audio_path'].apply(lambda x: os.path.basename(x).split('_')[0]) == uni_spkrs[idx]]
    wrds.append(set(temp_df['text']))

cmn_wrds = list(set.intersection(*map(set, wrds)))
print(len(cmn_wrds))
# print(cmn_wrds)

#########   To find OOV from cmn wrds of all spekrs

# df2 = pd.read_csv(sys.argv[2])

# uni_spkrs = df2['audio_path'].apply(lambda x: os.path.basename(x).split('_')[0]).unique()
# print(uni_spkrs)

# wrds_new = []
# for idx,spkr in enumerate(uni_spkrs):
#     # one speaker per set
#     temp_df = df2[df2['audio_path'].apply(lambda x: os.path.basename(x).split('_')[0]) == uni_spkrs[idx]]
#     wrds_new.append(set(temp_df['text']))

# cmn_wrds1 = list(set.intersection(*map(set, wrds_new)))
# oov = [item for item in cmn_wrds if item not in cmn_wrds1]
# print(f'OOV wrds ----> {oov}')