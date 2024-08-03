import pandas as pd
import numpy as np
import sys, os
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import root_mean_squared_error

# file = sys.argv[1] # path of pred/metrics file

fldr = sys.argv[1]

severity_dict = { 'verylow': ['F05','M08','M09','M10','M14'],
                'low' : ['F04','M05','M11'],
                'mid' : ['F02','M07','M16'],
                'high' : ['F03','M01','M04','M12']}
IP_dict = {'M04':[.02,'high'], 'F03':[.06,'high'], 'M12':[.074,'high'], 'M01':[.15,'high'], 
           'M07':[.28,'mid'], 'F02':[.29,'mid'],  'M16':[.43,'mid'], 
           'M05':[.58,'low'],  'M11':[.62,'low'], 'F04':[.62, 'low'],
           'M09':[.86,'verylow'], 'M14':[.904,'verylow'],  'M08':[.93,'verylow'], 'M10':[.93,'verylow'], 'F05':[.95, 'verylow']}


files = [i for i in os.listdir(fldr) if i.endswith('.csv')]
print(files)
for file in files:
    df= pd.read_csv(os.path.join(fldr,file))
    out_file = f'{os.path.join(fldr)}Severity_PC_{os.path.basename(os.path.dirname(fldr))}.txt'
    print(out_file)
    ##### find correlation co-eff
    with open (out_file, 'a') as f1:
        f1.write(f'Pearson, Spearman & RMSE for the file {file} \n')
        for key in severity_dict:
                print(key)
                spkr = severity_dict[key]
                df1 = df[df['spkrs'].isin(spkr)]  
                y_new = [IP_dict[i][0] for i in spkr]   
                zipped = list(zip(spkr, y_new))
                sorted_zipped = sorted(zipped, key=lambda x: x[1])
                # print(sorted_zipped)
                desired_order = [x[0] for x in sorted_zipped]
                ip_gvn = [x[1] for x in sorted_zipped]
                # print(ip_gvn)
                df1['spkrs'] = pd.Categorical(df1['spkrs'], categories=desired_order, ordered=True)
                df_sorted = df1.sort_values('spkrs')
                # df_sorted = df_sorted.reset_index(drop=True)
                # print(df_sorted.head())
                # sys.exit()
                for i in ['acc','avg_proba','positive_pred']:
                        pc = pearsonr(ip_gvn, df_sorted[i] )
                        sp = spearmanr(ip_gvn, df_sorted[i] )
                        rmse = root_mean_squared_error(ip_gvn, df_sorted[i])
                        # print(f'For {i} : {pc} \n')
                        # print(f'For {i} : {sp} \n')
                        # print(f'For {i} : {rmse} \n')
                        # sys.exit()
                        f1.write(f'For {i} & {key} : Pearson = {pc} \n')
                        f1.write(f'For {i} & {key}: Spearman = {sp} \n')
                        f1.write(f'For {i} & {key}: RMSE = {rmse} \n')
            
print('completted !!!!!')