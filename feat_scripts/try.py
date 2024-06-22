import sys
import subprocess

feat = ['melspec', 'mfcc', 'mfcc_del' ] #, 'plp_mfcc'] #, 'sdc_mel']

set = ['train', 'test', 'val' ] #, 320, 13]

for i in range(len(feat)):
    for j in range(len(set)):
        print(f'Extracting {feat[i]} for {set[j]}')
        args ='python feat_extract_all.py ' + feat[i] +  ' ' + set[j]
        subprocess.run( args, shell=True) 
    
print("Completted !!!")