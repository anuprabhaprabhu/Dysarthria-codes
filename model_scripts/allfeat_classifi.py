import sys
import subprocess

feat = ['melspec']  #, 'mfcc', 'mfcc_del', 'plp', 'sdc']
epoch = 100
feat_shape = [80 ]  #, 13, 26 , 13, 280]

for i in range(len(feat)):
    # print(feat[i], feat_shape[i])
    print("  ######################################################    ")
    print(f'Training on {feat[i]} feature for {epoch} epochs with feat dimention {feat_shape[i]}')
    args ='python classification_ua.py ' + feat[i] + ' ' + str(epoch) + ' ' + str(feat_shape[i])
    with open(f'jun22_logs.txt', 'a') as log_file:
        # subprocess.run( args, shell=True) 
        subprocess.run(args, shell=True, stdout=log_file, stderr=log_file)
    log_file.close()
        
print("Completed !!!")