import subprocess
import numpy as np
from python_speech_features import mfcc, delta
from python_speech_features import logfbank
import scipy.io.wavfile as wav
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import os
import time
import sys
import diffsptk

start = time.time()
    
def compute_delta(wav_file):
    (rate,sig) = wav.read(wav_file)
    mfcc_feat = mfcc(sig,rate)
    mfcc_delta = delta(mfcc_feat, 1)
    delta_feat = np.concatenate((mfcc_feat, mfcc_delta), axis=1)

    return delta_feat


def extract_features(csv_path, audio_path, feat_path, db_list):
    
    if not os.path.exists(feat_path):
       os.makedirs(feat_path)

    if 'google_v1' in audio_path.split('/')[-2] or 'qualcomm' in audio_path.split('/')[-2]:
        metadata = pd.read_csv(csv_path)
        unique_audio_path = metadata['audio_path'].unique()

    else:
        temp=[]
        #forming positive samples
        for db in db_list:
            for csv_file in (Path(csv_path).rglob('*' + db + '*')):
                temp_metadata = pd.read_csv(csv_file)
                for index,row in tqdm(temp_metadata.iterrows()):
                    temp.append({'audio_path':row['anchor'], 'text': row['anchor_text']})
                    temp.append({'audio_path':row['comparison'], 'text': row['comparison_text']})
        metadata = pd.DataFrame(temp)
        
        unique_audio_path = metadata['audio_path'].unique()

    for audio_filename in tqdm(unique_audio_path):
        try:
            filename, file_extension = os.path.splitext(audio_filename)
            delta_feat = compute_delta(os.path.join(audio_path,audio_filename))
            directory_path = os.path.join(feat_path, os.path.dirname(filename))
            if not os.path.exists(directory_path):
               os.makedirs(directory_path)
            np.savez(os.path.join(feat_path,filename), audio_feat=delta_feat)
        
        except Exception as e:
            with open(f'/scratch/{scratch_username}/feats/audio_mfcc_delta_feats/error.log','a') as f:
                f.write(f"Error in {filename} is {e}")
                f.write('\n')
                f.close()
            

if __name__=="__main__":

   s_username = sys.argv[1]

   username = os.environ.get('USER')
   scratch_username=s_username

   print('\nfeature extraction started')
   csv_path = [f'/home2/{username}/kesav/database/LibriPhrase/metadata/',
                      f'/home2/{username}/kesav/database/eval/metadata/',
                      f'/home2/{username}/kesav/dataset_scripts/google_metadata/google_v1_eval_metadata.csv',
                      f'/home2/{username}/kesav/dataset_scripts/qualcomm_metadata/qualcomm_eval_metadata.csv']

   audio_path = [f'/scratch/{scratch_username}/database/LibriPhrase_diffspk_all/',
                      f'/scratch/{scratch_username}/database/',
                      f'/scratch/{scratch_username}/database/google_v1/',
                      f'/scratch/{scratch_username}/database/qualcomm/']

   feat_path = [f'/scratch/{scratch_username}/feats/audio_mfcc_delta_feats/LibriPhrase_diffspk_all/',
                      f'/scratch/{scratch_username}/feats/audio_mfcc_delta_feats/train_500/',
                      f'/scratch/{scratch_username}/feats/audio_mfcc_delta_feats/google_v1/',
                      f'/scratch/{scratch_username}/feats/audio_mfcc_delta_feats/qualcomm/']


   db_list = [['train_100','train_360'], ['train_500'], ['train_500'], ['train_500']]

   for csv_path, audio_path, feat_path, db_list in zip(csv_path, audio_path, feat_path, db_list):
       extract_features(csv_path, audio_path, feat_path, db_list)
   
   print('\nfeature extraction completed')
  
end =  time.time()
duration=(end-start)/60
print(f"\nthe execution time is {duration} minutes")
