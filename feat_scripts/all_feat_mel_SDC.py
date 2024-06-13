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

start = time.time()
def sdc(cep, d, p, k):
    """


    :param cep: matrix of feature, 1 vector per line
    :param d: represents the time advance and delay for the delta computation
    :param k: number of delta-cepstral blocks whose delta-cepstral
       coefficients are stacked to form the final feature vector
    :param p: time shift between consecutive blocks.

    return: cepstral coefficient concatenated with shifted deltas
    """

    y = np.r_[np.resize(cep[0, :], (d, cep.shape[1])),
                 cep,
                 np.resize(cep[-1, :], (k * 3 + d, cep.shape[1]))]

    delta = compute_delta(y, win=d, method='diff')
    sdc = np.empty((cep.shape[0], cep.shape[1] * k))

    idx = np.zeros(delta.shape[0], dtype='bool')
    for ii in range(k):
        idx[d + ii * p] = True
    for ff in range(len(cep)):
        sdc[ff, :] = delta[idx, :].reshape(1, -1)
        idx = np.roll(idx, 1)
    return np.hstack((cep, sdc))

def compute_delta(features,
                  win=3,
                  method='filter',
                  filt=np.array([.25, .5, .25, 0, -.25, -.5, -.25])):
    """features is a 2D-ndarray  each row of features is a a frame

    :param features: the feature frames to compute the delta coefficients
    :param win: parameter that set the length of the computation window.
            The size of the window is (win x 2) + 1
    :param method: method used to compute the delta coefficients
        can be diff or filter
    :param filt: definition of the filter to use in "filter" mode, default one
        is similar to SPRO4:  filt=numpy.array([.2, .1, 0, -.1, -.2])

    :return: the delta coefficients computed on the original features.
    """
    # First and last features are appended to the begining and the end of the
    # stream to avoid border effect
    x = np.zeros((features.shape[0] + 2 * win, features.shape[1]), dtype=np.float32)
    x[:win, :] = features[0, :]
    x[win:-win, :] = features
    x[-win:, :] = features[-1, :]

    delta = np.zeros(x.shape, dtype=np.float32)

    if method == 'diff':
        filt = np.zeros(2 * win + 1, dtype=np.float32)
        filt[0] = -1
        filt[-1] = 1

    for i in range(features.shape[1]):
        delta[:, i] = np.convolve(features[:, i], filt)

    return delta[win:-win, :]

def extract_mfcc(wav_file, N):

    (rate,sig) = wav.read(wav_file)
    mfcc_feat = logfbank(sig,rate, nfilt=N)
    return mfcc_feat

def extract_features(csv_path, audio_path, feat_path, db_list, N, d, p, k):
    
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
            mfcc_feat = extract_mfcc(os.path.join(audio_path,audio_filename), N)
            sdc_feat = sdc(mfcc_feat, d, p, k)
            directory_path = os.path.join(feat_path, os.path.dirname(filename))
            if not os.path.exists(directory_path):
               os.makedirs(directory_path)
            np.savez(os.path.join(feat_path,filename), audio_feat=sdc_feat)
        
        except Exception as e:
            print(f"Error in {filename} is ", e)
            sys.exit()

if __name__=="__main__":

   N = int(sys.argv[1])
   d = int(sys.argv[2])
   p = int(sys.argv[3])
   k = int(sys.argv[4])
   s_username = sys.argv[5]

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

   feat_path = [f'/scratch/{scratch_username}/feats/audio_sdc_{N}_{d}_{p}_{k}_feats/LibriPhrase_diffspk_all/',
                      f'/scratch/{scratch_username}/feats/audio_sdc_{N}_{d}_{p}_{k}_feats/train_500/',
                      f'/scratch/{scratch_username}/feats/audio_sdc_{N}_{d}_{p}_{k}_feats/google_v1/',
                      f'/scratch/{scratch_username}/feats/audio_sdc_{N}_{d}_{p}_{k}_feats/qualcomm/']


   db_list = [['train_100','train_360'], ['train_500'], ['train_500'], ['train_500']]

   for csv_path, audio_path, feat_path, db_list in zip(csv_path, audio_path, feat_path, db_list):
       extract_features(csv_path, audio_path, feat_path, db_list, N, d, p, k)
   
   print('\nfeature extraction completed')
  
end =  time.time()
duration=(end-start)/60
print(f"\nthe execution time is {duration} minutes")
