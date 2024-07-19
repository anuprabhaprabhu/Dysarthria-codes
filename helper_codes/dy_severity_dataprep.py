import pandas as pd 
import os, sys

inp = sys.argv[1]
df = pd.read_csv(inp)

severity_dict = { 'verylow': ['M04','F03','M12'],
                    'low' : ['M07','F02','M16'],
                    'mid' : ['M05','F04','M11'],
                'high' : ['M09','M10','M14']}

for i,key in enumerate(severity_dict):
    spkr = severity_dict[key]
    new_df = df[df['audio_path'].apply(lambda x: os.path.basename(x).split('_')[0]).isin(spkr)]
    new_df['label'] = new_df['label'].map(lambda x: i)
    # print(new_df.head())
    # sys.exit()
    new_df.to_csv(os.path.join(os.path.dirname(inp),f'{key}_mfcc.csv'), sep=',', index=False)

print('Completted !!!!')