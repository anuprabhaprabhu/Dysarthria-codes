import pandas as pd 
import os, sys


fle = sys.argv[1]   # input csv file
tag = sys.argv[2]   # ctrl/dysar
df = pd.read_csv(fle)

uni_spkrs = df['audio_path'].apply(lambda x: os.path.basename(x).split('_')[0]).unique()
print(uni_spkrs)
# sys.exit()
digi = ['D0','D1','D2','D3','D4','D5','D6','D7','D8','D9']
ltrs = ['LA','LB','LC','LD','LE','LF','LG','LH','LI','LJ','LK','LL','LM','LN','LO','LP','LQ','LR','LS','LT','LU','LV','LW','LX','LY','LZ']
cmds = ['C10','C11','C12','C13','C14','C15','C16','C17','C18','C19','C1','C2','C3','C4','C5','C6','C7','C8','C9']

#### To get balanced dysarthria stats - 12 speakers
spkr_exclu = ['M01','M08']
if tag == 'dysar':
    df = df[~df['audio_path'].apply(lambda x: os.path.basename(x).split('_')[0]).isin(spkr_exclu)]

#### for the entire file
all_digi = pd.DataFrame()
all_ltrs = pd.DataFrame()
all_cmds = pd.DataFrame()
all_CW = pd.DataFrame()
all_UW = pd.DataFrame()
b1 = pd.DataFrame()
b2 = pd.DataFrame()
b3 = pd.DataFrame()

############## Speakerwise
for idx,spkr in enumerate(uni_spkrs):
    ####### Spkrwise
    temp_df = df[df['audio_path'].apply(lambda x: os.path.basename(x).split('_')[0]) == uni_spkrs[idx]]
    temp_df.to_csv(os.path.join(os.path.dirname(fle),f'{tag}/spkr_wise/{uni_spkrs[idx]}.csv'), sep=',', index=False)

    #####  Blockwise
    for j in ['B1','B2','B3']:
        blk_df = temp_df[temp_df['audio_path'].apply(lambda x: os.path.basename(x).split('_')[1])== j]
        blk_df.to_csv(os.path.join(os.path.dirname(fle),f'{tag}/{j}_spkr/{uni_spkrs[idx]}_{j}.csv'), sep=',', index=False)
        if j == 'B1':
            b1 = pd.concat([b1, blk_df], ignore_index=True)
        elif j == 'B2':
            b2 = pd.concat([b2, blk_df], ignore_index=True)
        elif j == 'B3':
            b3 = pd.concat([b3, blk_df], ignore_index=True)

    ######catagorywise
    ############   Digits
    digi_cnt = temp_df[temp_df['audio_path'].apply(lambda x: os.path.basename(x).split('_')[2]).isin(digi)]
    print(f'digits covered by spkr {uni_spkrs[idx]} ',digi_cnt['text'].nunique())
    digi_cnt.to_csv(os.path.join(os.path.dirname(fle),f'{tag}/digi_spkr/{uni_spkrs[idx]}.csv'), sep=',', index=False)
    all_digi = pd.concat([all_digi, digi_cnt], ignore_index=True)

    #######   Letters
    ltrs_cnt = temp_df[temp_df['audio_path'].apply(lambda x: os.path.basename(x).split('_')[2]).isin(ltrs)]
    print(f'Letters covered by spkr {uni_spkrs[idx]}', ltrs_cnt['text'].nunique())
    ltrs_cnt.to_csv(os.path.join(os.path.dirname(fle),f'{tag}/letr_spkr/{uni_spkrs[idx]}.csv'), sep=',', index=False)
    all_ltrs = pd.concat([all_ltrs, ltrs_cnt], ignore_index=True)

    #########  Commands
    cmds_cnt = temp_df[temp_df['audio_path'].apply(lambda x: os.path.basename(x).split('_')[2]).isin(cmds)]
    print(f'commands covered by spkr {uni_spkrs[idx]}', cmds_cnt['text'].nunique())
    cmds_cnt.to_csv(os.path.join(os.path.dirname(fle),f'{tag}/cmd_spkr/{uni_spkrs[idx]}.csv'), sep=',', index=False)
    all_cmds = pd.concat([all_cmds, cmds_cnt], ignore_index=True)

    #######  Common words
    cmnwrds_cnt = temp_df[temp_df['audio_path'].apply(lambda x: os.path.basename(x).split('_')[2][:2])=='CW']
    print(f'Common words covered by spkr {uni_spkrs[idx]}', cmnwrds_cnt['text'].nunique())
    cmnwrds_cnt.to_csv(os.path.join(os.path.dirname(fle),f'{tag}/cmn_spkr/{uni_spkrs[idx]}.csv'), sep=',', index=False)
    all_CW = pd.concat([all_CW, cmnwrds_cnt], ignore_index=True)

    ######## Uncommon words    
    uncmnwrds_cnt = temp_df[temp_df['audio_path'].apply(lambda x: os.path.basename(x).split('_')[2][:2])=='UW']
    print(f'uncommon words covered by spkr {uni_spkrs[idx]}', uncmnwrds_cnt['text'].nunique())
    uncmnwrds_cnt.to_csv(os.path.join(os.path.dirname(fle),f'{tag}/uncmn_spkr/{uni_spkrs[idx]}.csv'), sep=',', index=False)
    all_UW = pd.concat([all_UW, uncmnwrds_cnt], ignore_index=True)

#### combine category wise files 
if tag == 'ctrl':
    b1.to_csv(os.path.join(os.path.dirname(fle),f'{tag}/B1_spkr/B1.csv'), sep=',', index=False)
    b2.to_csv(os.path.join(os.path.dirname(fle),f'{tag}/B2_spkr/B2.csv'), sep=',', index=False)
    b3.to_csv(os.path.join(os.path.dirname(fle),f'{tag}/B3_spkr/B3.csv'), sep=',', index=False)
    all_digi.to_csv(os.path.join(os.path.dirname(fle),f'{tag}/digi_spkr/{tag}_digi.csv'), sep=',', index=False)
    all_ltrs.to_csv(os.path.join(os.path.dirname(fle),f'{tag}/letr_spkr/{tag}_letrs.csv'), sep=',', index=False)
    all_cmds.to_csv(os.path.join(os.path.dirname(fle),f'{tag}/cmd_spkr/{tag}_cmd.csv'), sep=',', index=False)
    all_CW.to_csv(os.path.join(os.path.dirname(fle),f'{tag}/cmn_spkr/{tag}_cmn_wrds.csv'), sep=',', index=False)
    all_UW.to_csv(os.path.join(os.path.dirname(fle),f'{tag}/uncmn_spkr/{tag}_uncmnwrds.csv'), sep=',', index=False)

if tag == 'dysar':
    b1.to_csv(os.path.join(os.path.dirname(fle),f'{tag}/B1_spkr/B1.csv'), sep=',', index=False)
    b2.to_csv(os.path.join(os.path.dirname(fle),f'{tag}/B2_spkr/B2.csv'), sep=',', index=False)
    b3.to_csv(os.path.join(os.path.dirname(fle),f'{tag}/B3_spkr/B3.csv'), sep=',', index=False)
    all_digi.to_csv(os.path.join(os.path.dirname(fle),f'{tag}/digi_spkr/{tag}_digi.csv'), sep=',', index=False)
    all_ltrs.to_csv(os.path.join(os.path.dirname(fle),f'{tag}/letr_spkr/{tag}_letrs.csv'), sep=',', index=False)
    all_cmds.to_csv(os.path.join(os.path.dirname(fle),f'{tag}/cmd_spkr/{tag}_cmd.csv'), sep=',', index=False)
    all_CW.to_csv(os.path.join(os.path.dirname(fle),f'{tag}/cmn_spkr/{tag}_cmn_wrds.csv'), sep=',', index=False)
    all_UW.to_csv(os.path.join(os.path.dirname(fle),f'{tag}/uncmn_spkr/{tag}_uncmnwrds.csv'), sep=',', index=False)
    ###### to get severity based split
    ########  dictionary of severity level and speakers
    severity_dict = { 'verylow': ['M04','F03','M12'],
                     'low' : ['M07','F02','M16'],
                     'mid' : ['M05','F04','M11'],
                    'high' : ['M09','M10','M14']}
    for key in severity_dict:
        spkr = severity_dict[key]
        
        new_b1 = b1[b1['audio_path'].apply(lambda x: os.path.basename(x).split('_')[0]).isin(spkr)]
        new_b1.to_csv(os.path.join(os.path.dirname(fle),f'{tag}/B1_spkr/{key}_b1.csv'), sep=',', index=False)

        new_b2 = b2[b2['audio_path'].apply(lambda x: os.path.basename(x).split('_')[0]).isin(spkr)]
        new_b2.to_csv(os.path.join(os.path.dirname(fle),f'{tag}/B2_spkr/{key}_b2.csv'), sep=',', index=False)

        new_b3 = b3[b3['audio_path'].apply(lambda x: os.path.basename(x).split('_')[0]).isin(spkr)]
        b3.to_csv(os.path.join(os.path.dirname(fle),f'{tag}/B3_spkr/{key}_b3.csv'), sep=',', index=False)

        new_all_digi = all_digi[all_digi['audio_path'].apply(lambda x: os.path.basename(x).split('_')[0]).isin(spkr)]
        new_all_digi.to_csv(os.path.join(os.path.dirname(fle),f'{tag}/digi_spkr/{key}_digi.csv'), sep=',', index=False)

        new_all_ltrs = all_ltrs[all_ltrs['audio_path'].apply(lambda x: os.path.basename(x).split('_')[0]).isin(spkr)]
        new_all_ltrs.to_csv(os.path.join(os.path.dirname(fle),f'{tag}/letr_spkr/{key}_letrs.csv'), sep=',', index=False)

        new_all_cmds = all_cmds[all_cmds['audio_path'].apply(lambda x: os.path.basename(x).split('_')[0]).isin(spkr)]
        new_all_cmds.to_csv(os.path.join(os.path.dirname(fle),f'{tag}/cmd_spkr/{key}_cmd.csv'), sep=',', index=False)

        new_all_CW = all_CW[all_CW['audio_path'].apply(lambda x: os.path.basename(x).split('_')[0]).isin(spkr)]
        new_all_CW.to_csv(os.path.join(os.path.dirname(fle),f'{tag}/cmn_spkr/{key}_cmn_wrds.csv'), sep=',', index=False)
        
        new_UW = all_UW[all_UW['audio_path'].apply(lambda x: os.path.basename(x).split('_')[0]).isin(spkr)]
        new_UW.to_csv(os.path.join(os.path.dirname(fle),f'{tag}/uncmn_spkr/{key}_UW.csv'), sep=',', index=False)

        new_df = df[df['audio_path'].apply(lambda x: os.path.basename(x).split('_')[0]).isin(spkr)]
        new_df.to_csv(os.path.join(os.path.dirname(fle),f'{tag}/spkr_wise/{key}.csv'), sep=',', index=False)

print("Completed !!!!!!!!!!")