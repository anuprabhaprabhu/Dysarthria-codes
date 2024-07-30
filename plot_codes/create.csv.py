import pandas as pd
import numpy as np
import sys, os


file = sys.argv[1] # path of pred/metrics file

def create_df(inp):
    #### process the pred file
    df= pd.read_csv(inp, sep=",", header=None, names= ['spkrs','acc','F1_score','avg_proba','y_pre_avg'])
    # print(df.head())
    # sys.exit()
    df['spkrs'] = df['spkrs'].str.replace('.csv','')
    df['spkrs'] = df['spkrs'].str.replace('_CW','')
    df['spkrs'] = df['spkrs'].str.replace('file_name=','')
    df['acc'] = df['acc'].str.replace('acc=','')
    df['F1_score'] = df['F1_score'].str.replace('f1_score=','')
    df['avg_proba'] = df['avg_proba'].str.replace('avg_prob=','')
    df['acc'] = df['acc'].map(lambda x: float(x))
    df['F1_score'] = df['F1_score'].map(lambda x: float(x))
    df['avg_proba'] = df['avg_proba'].map(lambda x: float(x))
    # print(df.head())
    # df = df[df['spkrs'] != 'F05']
    # df = df[df['spkrs'] != 'M01']
    # df = df[df['spkrs'] != 'M08']
    # df = df.sort_values(by='avg_proba')
    return df

df1 = create_df(file)
df1.to_csv(file.replace('.txt','.csv'), sep =',', index =False)
print('Completed !!!!')