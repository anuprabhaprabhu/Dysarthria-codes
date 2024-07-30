import pandas as pd
import numpy as np
import sys
# matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
import re 
from scipy.stats import norm
from scipy import stats

inp = sys.argv[1] # path of pred/metrics file

#### process the pred file
df= pd.read_csv(inp)

IP_dict = {'M04':.02, 'F03':.06, 'M12':.074, 'M01':.15, 'M07':.28, 'F02':.29, 'M16':.43,'M05':.58, 
        'M11':.62,'F04':.62, 'M09':.86,'M14':.904,  'M08':.93, 'M10':.93,'F05':.95}
IP_dict = dict(sorted(IP_dict.items(), key=lambda item: item[1]))

# df['spkrs'] = df['spkrs'].str.replace('.csv','')
# df['spkrs'] = df['spkrs'].str.replace('_CW','')
# df['spkrs'] = df['spkrs'].str.replace('file_name=','')
# df['acc'] = df['acc'].str.replace('acc=','')
# df['F1_score'] = df['F1_score'].str.replace('f1_score=','')
# df['avg_proba'] = df['avg_proba'].str.replace('avg_prob=','')
# df['acc'] = df['acc'].map(lambda x: float(x))
# df['F1_score'] = df['F1_score'].map(lambda x: float(x))
# df['avg_proba'] = df['avg_proba'].map(lambda x: float(x))
# # print(df.head())
# # df = df[df['spkrs'] != 'F05']
# # df = df[df['spkrs'] != 'M01']
# # df = df[df['spkrs'] != 'M08']
# # df = df.sort_values(by='avg_proba')

order_df = pd.DataFrame(list(IP_dict.items()), columns=['Key', 'Order'])
order_df.set_index('Key', inplace=True)
df['Order'] = df['spkrs'].map(order_df['Order'])
df = df.sort_values(by='Order').drop(columns=['Order'])


# ##### plots
# # plt.bar(df['spkrs'], df['acc'])
# # plt.title('Accuracy plot for dysarthria')
# # plt.xlabel('Dysarthria speakers')
# # plt.ylabel('Accuracy')

x = list(IP_dict.keys())  # Dates or categories on x-axis
y = list(IP_dict.values())
plt.plot(x,y, marker='s',linestyle='-', color='k', label= ' IP_values')
plt.plot(x,df['acc'], marker='s', color='r', label= 'acc')
plt.plot(x,df['avg_proba'], marker='s', color='g', label= 'avg_proba')
plt.plot(x,df['positive_pred'], marker='s', color='y', label= 'positive_pred')
plt.title('avg_proba plot for dysarthria')
plt.xlabel('Dysarthria speakers')
plt.ylabel('avg_proba & IP')
plt.legend()
plt.show()