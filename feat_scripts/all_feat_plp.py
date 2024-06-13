import subprocess
import numpy as np
from python_speech_features import mfcc
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
    
def extract_plp(wav_file, fl, fp, n_fft, n_channel, M):

    x, sr = diffsptk.read(wav_file)

    stft = diffsptk.STFT(frame_length=fl, frame_period=fp, fft_length=n_fft)

    X = stft(x)
    # Extract PLP.
    plp = diffsptk.PLP(
        plp_order=M,
        n_channel=n_channel,
        fft_length=n_fft,
        sample_rate=sr,
    )
    plp_feat = plp(X)
    plp_feat_numpy = plp_feat.numpy()
    #(rate,sig) = wav.read(wav_file)
    #plps  = plp(sig)
    
    return plp_feat_numpy

def extract_features(csv_path, audio_path, feat_path, db_list, fl, fp, n_fft, n_channel, M):
    
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
            plp_feat = extract_plp(os.path.join(audio_path,audio_filename), fl, fp, n_fft, n_channel, M)
            directory_path = os.path.join(feat_path, os.path.dirname(filename))
            if not os.path.exists(directory_path):
               os.makedirs(directory_path)
            np.savez(os.path.join(feat_path,filename), audio_feat=plp_feat)
        
        except Exception as e:
            with open(f'/scratch/{scratch_username}/feats/audio_plp_feats/error.log','a') as f:
                f.write(f"Error in {filename} is {e}")
                f.write('\n')
                f.close()
            

if __name__=="__main__":

   s_username = sys.argv[1]

   fl = 400        # Frame length
   fp = 160         # Frame period
   n_fft = 512     # FFT length
   n_channel = 80  # Number of channels
   M = 13          # MFCC/PLP dimensions
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

   feat_path = [f'/scratch/{scratch_username}/feats/audio_plp_feats/LibriPhrase_diffspk_all/',
                      f'/scratch/{scratch_username}/feats/audio_plp_feats/train_500/',
                      f'/scratch/{scratch_username}/feats/audio_plp_feats/google_v1/',
                      f'/scratch/{scratch_username}/feats/audio_plp_feats/qualcomm/']


   db_list = [['train_100','train_360'], ['train_500'], ['train_500'], ['train_500']]

   for csv_path, audio_path, feat_path, db_list in zip(csv_path, audio_path, feat_path, db_list):
       extract_features(csv_path, audio_path, feat_path, db_list, fl, fp, n_fft, n_channel, M)
   
   print('\nfeature extraction completed')
  
end =  time.time()
duration=(end-start)/60
print(f"\nthe execution time is {duration} minutes")
