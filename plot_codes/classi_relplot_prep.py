import sys, os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

fldr_pth = sys.argv[1]

severity_dict = { 'verylow': ['F05','M08','M09','M10','M14'],
                    'low' : ['M05','F04','M11'],
                    'mid' : ['F02','M07','M16'],
                   'high' : ['F03','M01','M04','M12']}

speaker_to_class = {speaker: cls for cls, speakers in severity_dict.items() for speaker in speakers}

print(speaker_to_class)

df = pd.DataFrame()

df1 = pd.read_csv(os.path.join(fldr_pth,'classi_char1_sid_spkr_digi.csv'))
df1['class'] = df1['spkrs'].map(speaker_to_class)
df1['groups'] = 'D'
df = pd.concat([df, df1], ignore_index=True)

df1 = pd.read_csv(os.path.join(fldr_pth,'classi_char1_sid_spkr_cmds.csv'))
df1['class'] = df1['spkrs'].map(speaker_to_class)
df1['groups'] = 'C'
df = pd.concat([df, df1], ignore_index=True)

df1 = pd.read_csv(os.path.join(fldr_pth,'classi_char1_sid_spkr_ltrs.csv'))
df1['class'] = df1['spkrs'].map(speaker_to_class)
df1['groups'] = 'L'
df = pd.concat([df, df1], ignore_index=True)

df1 = pd.read_csv(os.path.join(fldr_pth,'classi_char1_sid_spkr_CW.csv'))
df1['class'] = df1['spkrs'].map(speaker_to_class)
df1['groups'] = 'CW'
df = pd.concat([df, df1], ignore_index=True)

df1 = pd.read_csv(os.path.join(fldr_pth,'classi_char1_sid_spkr_UW.csv'))
df1['class'] = df1['spkrs'].map(speaker_to_class)
df1['groups'] = 'UW'
df = pd.concat([df, df1], ignore_index=True)

df1 = pd.read_csv(os.path.join(fldr_pth,'classi_char1_sid_spkr_B1.csv'))
df1['class'] = df1['spkrs'].map(speaker_to_class)
df1['groups'] = 'B1'
df = pd.concat([df, df1], ignore_index=True)

df1 = pd.read_csv(os.path.join(fldr_pth,'classi_char1_sid_spkr_B2.csv'))
df1['class'] = df1['spkrs'].map(speaker_to_class)
df1['groups'] = 'B2'
df = pd.concat([df, df1], ignore_index=True)

df1 = pd.read_csv(os.path.join(fldr_pth,'classi_char1_sid_spkr_B3.csv'))
df1['class'] = df1['spkrs'].map(speaker_to_class)
df1['groups'] = 'B3'
df = pd.concat([df, df1], ignore_index=True)

df1 = pd.read_csv(os.path.join(fldr_pth,'classi_char1_sid_spkr_B1_UW.csv'))
df1['class'] = df1['spkrs'].map(speaker_to_class)
df1['groups'] = 'B1_UW'
df = pd.concat([df, df1], ignore_index=True)

df1 = pd.read_csv(os.path.join(fldr_pth,'classi_char1_sid_spkr_B2_UW.csv'))
df1['class'] = df1['spkrs'].map(speaker_to_class)
df1['groups'] = 'B2_UW'
df = pd.concat([df, df1], ignore_index=True)

df1 = pd.read_csv(os.path.join(fldr_pth,'classi_char1_sid_spkr_B3_UW.csv'))
df1['class'] = df1['spkrs'].map(speaker_to_class)
df1['groups'] = 'B3_UW'
df = pd.concat([df, df1], ignore_index=True)

df1 = pd.read_csv(os.path.join(fldr_pth,'classi_char1_sid_spkr_spkr.csv'))
df1['class'] = df1['spkrs'].map(speaker_to_class)
df1['groups'] = 'all words'
df = pd.concat([df, df1], ignore_index=True)
df.to_csv(os.path.join(fldr_pth,'all_df_char1_classi_nodupli.csv'), sep =',', index =False)

print("Completted !!!!")