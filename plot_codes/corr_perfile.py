import pandas as pd
import numpy as np
import sys, os
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import root_mean_squared_error

# file = sys.argv[1] # path of pred/metrics file

fldr = sys.argv[1]

def create_df(inp):
    print(inp)
    #### process the pred file
    df= pd.read_csv(inp, sep=",", header=None, names= ['spkrs','acc','F1_score','avg_proba','positive_pred'])
    df['spkrs'] = df['spkrs'].str.replace('.csv','')
    df['spkrs'] = df['spkrs'].str.replace('file_name=','')
    df['spkrs'] = df['spkrs'].str.replace('_digi','')
    df['spkrs'] = df['spkrs'].str.replace('_ltrs','')
    df['spkrs'] = df['spkrs'].str.replace('_cmds','')
    df['spkrs'] = df['spkrs'].str.replace('_B1','')
    df['spkrs'] = df['spkrs'].str.replace('_B2','')
    df['spkrs'] = df['spkrs'].str.replace('_B3','')
    df['spkrs'] = df['spkrs'].str.replace('_CW','')
    df['spkrs'] = df['spkrs'].str.replace('_UW','')
    df['acc'] = df['acc'].str.replace('acc=','')
    df['F1_score'] = df['F1_score'].str.replace('f1_score=','')
    df['avg_proba'] = df['avg_proba'].str.replace('avg_prob=','')
    df['positive_pred'] = df['positive_pred'].str.replace('positive_pred=','')


    df['acc'] = df['acc'].map(lambda x: float(x))
    df['F1_score'] = df['F1_score'].map(lambda x: float(x))
    df['avg_proba'] = df['avg_proba'].map(lambda x: float(x))
    df['positive_pred'] = df['positive_pred'].map(lambda x: float(x))
    return df


IP_dict = {'M04':.02, 'F03':.06, 'M12':.074, 'M01':.15, 'M07':.28, 'F02':.29, 'M16':.43,'M05':.58, 
        'M11':.62,'F04':.62, 'M09':.86,'M14':.904,  'M08':.93, 'M10':.93,'F05':.95}
# IP_dict = {'F02':.29,'F03':.06,'F04':.62, 'M04':.02,'M05':.58, 
#            'M07':.28,'M09':.86,'M10':.93,'M11':.62,'M12':.074,'M14':.904,'M16':.43}
IP_dict = dict(sorted(IP_dict.items(), key=lambda item: item[1]))
x = list(IP_dict.keys())  # categories on x-axis

y = list(IP_dict.values())

##   main 
files = [i for i in os.listdir(fldr) if i.endswith('.txt')]
print(files)
for file in files:
    df1 = create_df(os.path.join(fldr,file))
    df1.to_csv(os.path.join(fldr,file).replace('.txt','.csv'), sep =',', index =False)

    order_df = pd.DataFrame(list(IP_dict.items()), columns=['Key', 'Order'])
    order_df.set_index('Key', inplace=True)
    df1['Order'] = df1['spkrs'].map(order_df['Order'])
    df1 = df1.sort_values(by='Order').drop(columns=['Order'])

    out_file = f'{os.path.join(fldr)}PC_{os.path.basename(os.path.dirname(fldr))}.txt'
    print(out_file)
    ##### find correlation co-eff
    with open (out_file, 'a') as f1:
        f1.write(f'Pearson, Spearman & RMSE for the file {file} \n')
        for i in ['acc','avg_proba','positive_pred']:
            pc = pearsonr(y, df1[i] )
            sp = spearmanr(y, df1[i] )
            rmse = root_mean_squared_error(y, df1[i])
            # print(f'For {i} : {pc} \n')
            # print(f'For {i} : {sp} \n')
            # print(f'For {i} : {rmse} \n')
            # sys.exit()
            f1.write(f'For {i} : Pearson = {pc} \n')
            f1.write(f'For {i} : Spearman = {sp} \n')
            f1.write(f'For {i} : RMSE = {rmse} \n')
            

print('Completed !!!!')