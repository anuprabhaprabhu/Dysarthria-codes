import pandas as pd
import numpy as np
import sys
# %matplotlib inline
import matplotlib.pyplot as plt
# import seaborn as sns
import re 
from scipy.stats import norm
from scipy.stats import pearsonr


# inp1 = sys.argv[1] # path of pred/metrics file
# inp2 = sys.argv[2]
# inp3 = sys.argv[3]
# inp4 = sys.argv[4]
# inp5 = sys.argv[5]
# inp6 = sys.argv[6]

IP_dict = {'M04':.02, 'F03':.06, 'M12':.074, 'M01':.15, 'M07':.28, 'F02':.29, 'M16':.43,'M05':.58, 
           'M11':.62,'F04':.62, 'M09':.86,'M14':.904,  'M08':.93, 'M10':.93,'F05':.95}
# IP_dict = {'F02':.29,'F03':.06,'F04':.62, 'M04':.02,'M05':.58, 
#            'M07':.28,'M09':.86,'M10':.93,'M11':.62,'M12':.074,'M14':.904,'M16':.43}
IP_dict = dict(sorted(IP_dict.items(), key=lambda item: item[1]))
x = list(IP_dict.keys())  # Dates or categories on x-axis
y = list(IP_dict.values())

df_spkr = pd.read_csv('/home/anuprabha/Desktop/anu_donot_touch/code/model_scripts/pred_dysar_6sec/by_mdlspdpertb_6sec/spkr_wise.csv')
df_cmn = pd.read_csv('/home/anuprabha/Desktop/anu_donot_touch/code/model_scripts/pred_dysar_6sec/by_mdlspdpertb_6sec/cmnwrds_spkrwise.csv')
df_uncmn = pd.read_csv('/home/anuprabha/Desktop/anu_donot_touch/code/model_scripts/pred_dysar_6sec/by_mdlspdpertb_6sec/uncmnwrds_spkrwise.csv')
df_B1 = pd.read_csv('/home/anuprabha/Desktop/anu_donot_touch/code/model_scripts/pred_dysar_6sec/by_mdlspdpertb_6sec/B1_spkrwise.csv')
df_B2 = pd.read_csv('/home/anuprabha/Desktop/anu_donot_touch/code/model_scripts/pred_dysar_6sec/by_mdlspdpertb_6sec/B2_spkrwise.csv')
df_B3 = pd.read_csv('/home/anuprabha/Desktop/anu_donot_touch/code/model_scripts/pred_dysar_6sec/by_mdlspdpertb_6sec/B3_spkrwise.csv')

order_df = pd.DataFrame(list(IP_dict.items()), columns=['Key', 'Order'])
order_df.set_index('Key', inplace=True)
df_spkr['Order'] = df_spkr['spkrs'].map(order_df['Order'])
df_spkr = df_spkr.sort_values(by='Order').drop(columns=['Order'])
df_cmn['Order'] = df_cmn['spkrs'].map(order_df['Order'])
df_cmn = df_cmn.sort_values(by='Order').drop(columns=['Order'])
df_uncmn['Order'] = df_uncmn['spkrs'].map(order_df['Order'])
df_uncmn = df_uncmn.sort_values(by='Order').drop(columns=['Order'])
df_B1['Order'] = df_B1['spkrs'].map(order_df['Order'])
df_B1 = df_B1.sort_values(by='Order').drop(columns=['Order'])
df_B2['Order'] = df_B2['spkrs'].map(order_df['Order'])
df_B2 = df_B2.sort_values(by='Order').drop(columns=['Order'])
df_B3['Order'] = df_B3['spkrs'].map(order_df['Order'])
df_B3 = df_B3.sort_values(by='Order').drop(columns=['Order'])


##### find correlation co-eff
pc = pearsonr(y, df_B3['avg_proba'] )
print(pc)

# plt.scatter(df_spkr['avg_proba'], y, color='red', marker='x')
# plt.show()
sys.exit()

##### plots
# plt.bar(df['spkrs'], df['acc'])
# plt.title('Accuracy plot for dysarthria')
# plt.xlabel('Dysarthria speakers')
# plt.ylabel('Accuracy')


plt.plot(x,y, marker='s',linestyle='-', color='k', label= ' IP_values')
plt.plot(x,df_spkr['avg_proba'], marker='s', color='r', label= '6sec_8spkr')
plt.plot(x,df_cmn['avg_proba'], marker='s', color='g', label= '6sec_spdpertb')
plt.plot(x,df_uncmn['avg_proba'], marker='s', color='b', label= '6sec_new')
plt.plot(x,df_B1['avg_proba'], marker='s', color='m', label= '10sec_8spkr')
plt.plot(x,df_B2['avg_proba'], marker='s', color='y', label= '10sec_spdpertb')
plt.plot(x,df_B3['avg_proba'], marker='s', color='c', label= '10sec_new')
plt.title('avg_proba plot for dysarthria')
plt.xlabel('Dysarthria speakers')
plt.ylabel('avg_proba & IP based on Intelligibility score')
plt.legend()
# plt.show()