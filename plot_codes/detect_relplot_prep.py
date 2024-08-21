import sys, os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

fldr_pth = sys.argv[1]
ctrl_spkrs = ['CF02', 'CF03', 'CF04', 'CM04' ,'CM05', 'CM06', 'CM08', 'CM09', 'CM10', 'CM12','CM13' ]
dysar_spkrs = ['F02', 'F03', 'F04', 'M01', 'M04', 'M05', 'M07', 'M08', 'M09', 'M10', 'M11',
                'M12', 'M14', 'M16' ,'F05']


# files = [i for i in os.listdir(fldr_pth) if i.endswith('.txt')]
# print(files)
# sys.exit()

df = pd.DataFrame()

df1 = pd.read_csv(os.path.join(fldr_pth,'detec_char1_sid_spkr__digi.csv'))
df1['class'] = df1['spkrs'].apply(lambda x: 'healthy' if x in ctrl_spkrs else 'dysarthria')
df1['groups'] = 'D'
df = pd.concat([df, df1], ignore_index=True)

df1 = pd.read_csv(os.path.join(fldr_pth,'detec_char1_sid_spkr__cmds.csv'))
df1['class'] = df1['spkrs'].apply(lambda x: 'healthy' if x in ctrl_spkrs else 'dysarthria')
df1['groups'] = 'C'
df = pd.concat([df, df1], ignore_index=True)

df1 = pd.read_csv(os.path.join(fldr_pth,'detec_char1_sid_spkr__ltrs.csv'))
df1['class'] = df1['spkrs'].apply(lambda x: 'healthy' if x in ctrl_spkrs else 'dysarthria')
df1['groups'] = 'L'
df = pd.concat([df, df1], ignore_index=True)

df1 = pd.read_csv(os.path.join(fldr_pth,'detec_char1_sid_spkr__CW.csv'))
df1['class'] = df1['spkrs'].apply(lambda x: 'healthy' if x in ctrl_spkrs else 'dysarthria')
df1['groups'] = 'CW'
df = pd.concat([df, df1], ignore_index=True)

df1 = pd.read_csv(os.path.join(fldr_pth,'detec_char1_sid_spkr__UW.csv'))
df1['class'] = df1['spkrs'].apply(lambda x: 'healthy' if x in ctrl_spkrs else 'dysarthria')
df1['groups'] = 'UW'
df = pd.concat([df, df1], ignore_index=True)

df1 = pd.read_csv(os.path.join(fldr_pth,'detec_char1_sid_spkr__B1.csv'))
df1['class'] = df1['spkrs'].apply(lambda x: 'healthy' if x in ctrl_spkrs else 'dysarthria')
df1['groups'] = 'B1'
df = pd.concat([df, df1], ignore_index=True)

df1 = pd.read_csv(os.path.join(fldr_pth,'detec_char1_sid_spkr__B2.csv'))
df1['class'] = df1['spkrs'].apply(lambda x: 'healthy' if x in ctrl_spkrs else 'dysarthria')
df1['groups'] = 'B2'
df = pd.concat([df, df1], ignore_index=True)

df1 = pd.read_csv(os.path.join(fldr_pth,'detec_char1_sid_spkr__B3.csv'))
df1['class'] = df1['spkrs'].apply(lambda x: 'healthy' if x in ctrl_spkrs else 'dysarthria')
df1['groups'] = 'B3'
df = pd.concat([df, df1], ignore_index=True)

df1 = pd.read_csv(os.path.join(fldr_pth,'detec_char1_sid_spkr__B1_UW.csv'))
df1['class'] = df1['spkrs'].apply(lambda x: 'healthy' if x in ctrl_spkrs else 'dysarthria')
df1['groups'] = 'B1_UW'
df = pd.concat([df, df1], ignore_index=True)

df1 = pd.read_csv(os.path.join(fldr_pth,'detec_char1_sid_spkr__B2_UW.csv'))
df1['class'] = df1['spkrs'].apply(lambda x: 'healthy' if x in ctrl_spkrs else 'dysarthria')
df1['groups'] = 'B2_UW'
df = pd.concat([df, df1], ignore_index=True)

df1 = pd.read_csv(os.path.join(fldr_pth,'detec_char1_sid_spkr__B3_UW.csv'))
df1['class'] = df1['spkrs'].apply(lambda x: 'healthy' if x in ctrl_spkrs else 'dysarthria')
df1['groups'] = 'B3_UW'
df = pd.concat([df, df1], ignore_index=True)

df1 = pd.read_csv(os.path.join(fldr_pth,'detec_char1_sid_spkr_spkr_wise.csv'))
df1['class'] = df1['spkrs'].apply(lambda x: 'healthy' if x in ctrl_spkrs else 'dysarthria')
df1['groups'] = 'all words'
df = pd.concat([df, df1], ignore_index=True)
df.to_csv(os.path.join(fldr_pth,'all_df_char1.csv'), sep =',', index =False)

# plt.scatter(df1['class'][:11], df1['avg_proba'][:11], color='r', alpha=0.7, marker='o', s=100)
# plt.scatter(df1['class'][11:], df1['avg_proba'][11:], color='b', alpha=0.7, marker='o', s=100)
# sns.set_palette("bright")
# sns.relplot(data=df, x="groups", y="avg_proba", hue="class", marker='H')
# plt.title('Fig 1. Average probability of each speaker for different word groups. D-digits, C-computer commands,\
#             L - letters, CW - common words, UW - uncommon words, B1, B2, B3 - each blocks')
# # plt.legend(title='')
# plt.xlabel('Word groups')
# plt.ylabel('average probability')
# plt.show()