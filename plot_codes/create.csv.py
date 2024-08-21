import pandas as pd
import numpy as np
import sys, os


# file = sys.argv[1] # path of pred/metrics file

fldr_path = sys.argv[1]  # Replace 'your_file.txt' with the path to your actual file
files = [i for i in os.listdir(fldr_path) if i.endswith('.txt')]


def create_df(inp):
    #### process the pred file
    df= pd.read_csv(inp, sep=",", header=None, names= ['spkrs','acc','F1_score','avg_proba'])  #,'y_pre_avg'])
    # print(df.head())
    # sys.exit()
    df['spkrs'] = df['spkrs'].str.replace('.csv','')
    df['spkrs'] = df['spkrs'].str.replace('file_name=','')
    df['spkrs'] = df['spkrs'].str.replace('_digi','')
    df['spkrs'] = df['spkrs'].str.replace('_cmds','')
    df['spkrs'] = df['spkrs'].str.replace('_ltrs','')
    df['spkrs'] = df['spkrs'].str.replace('_CW','')
    df['spkrs'] = df['spkrs'].str.replace('_UW','')
    df['spkrs'] = df['spkrs'].str.replace('_B1','')
    df['spkrs'] = df['spkrs'].str.replace('_B2','')
    df['spkrs'] = df['spkrs'].str.replace('_B3','')
    df['spkrs'] = df['spkrs'].str.replace('_B1_UW','')
    df['spkrs'] = df['spkrs'].str.replace('_B2_UW','')
    df['spkrs'] = df['spkrs'].str.replace('_B3_UW','')
    df['spkrs'] = df['spkrs'].str.replace('_spkr','')

    df['acc'] = df['acc'].str.replace('acc=','')
    # df['F1_score'] = df['F1_score'].str.replace('f1_score=','')
    df['avg_proba'] = df['avg_proba'].str.replace('avg_prob=','')
    df['avg_proba'] = df['avg_proba'].str.replace('[','')
    df['avg_proba'] = df['avg_proba'].str.replace(']','')
    df['acc'] = df['acc'].map(lambda x: float(x))
    # df['F1_score'] = df['F1_score'].map(lambda x: float(x))
    df['avg_proba'] = df['avg_proba'].map(lambda x: float(x))

    
    # print(df.head())
    # df = df[df['spkrs'] != 'F05']
    # df = df[df['spkrs'] != 'M01']
    # df = df[df['spkrs'] != 'M08']
    # df = df.sort_values(by='avg_proba')
    return df

for file in files:
    file_path = os.path.join(fldr_path,file)
    df1 = create_df(file_path)
    df1.to_csv(file_path.replace('.txt','.csv'), sep =',', index =False)
print('Completed !!!!')