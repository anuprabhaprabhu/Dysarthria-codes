import pandas as pd
from python_speech_features import logfbank
from python_speech_features import mfcc, delta

import scipy.io.wavfile as wav 
import os 
import sys
import librosa
from tqdm import tqdm
import numpy as np


def mel_spec(wav_file):
    (rate,sig) = wav.read(wav_file)
    # mel_feat = logfbank(sig,rate, nfilt=40)
    mel_feat = logfbank(sig,rate, nfilt=80)

    return mel_feat

def extract_features():
    pass

def feat_mfcc(wav_file):
    (rate,sig) = wav.read(wav_file)
    mfcc_feat = mfcc(sig,rate)
    # mfcc_delta = delta(mfcc_feat, 1)
    # delta_feat = np.concatenate((mfcc_feat, mfcc_delta), axis=1)
    return mfcc_feat

def feat_mfcc_delta(wav_file):
    (rate,sig) = wav.read(wav_file)
    mfcc_feat = mfcc(sig,rate)
    mfcc_delta = delta(mfcc_feat, 1)
    delta_feat = np.concatenate((mfcc_feat, mfcc_delta), axis=1)
    return delta_feat

def feat_mfcc_delta_delta(wav_file):
    (rate,sig) = wav.read(wav_file)
    mfcc_feat = mfcc(sig,rate)
    mfcc_delta = delta(mfcc_feat, 1)
    mfcc_delta2 = delta(mfcc_delta, 1)
    delta_feat = np.concatenate((mfcc_feat, mfcc_delta, mfcc_delta2), axis=1)
    return delta_feat



########################
if __name__ == '__main__':
    feat = sys.argv[1]
    set = sys.argv[2]
    data_pth = '/home/anuprabha/Desktop/anu_donot_touch/data/'
    feat_pth =f'/home/anuprabha/Desktop/anu_donot_touch/feats/uaspeech/LOSO_spkr_feats/{feat}/'
    feat_sub_path = f'/uaspeech/LOSO_spkr_feats/{feat}/'
    metadata_path = f'/home/anuprabha/Desktop/anu_donot_touch/data/UASpeech/UASpeech_noisereduce/classification/LOSO_spkr/set_0/{set}_set_0.csv'

    #for PLP
    fl = 400        # Frame length
    fp = 160         # Frame period
    n_fft = 512     # FFT length
    n_channel = 80  # Number of channels
    M = 13          # MFCC/PLP dimensions

    metadata = pd.read_csv(metadata_path)
    feat_info = []

    for index,line  in tqdm(metadata.iterrows()):
        audio_filename = os.path.join(data_pth,line['audio_path'])#.replace('.wav','')
        new_audio_path = os.path.join(feat_pth, *audio_filename.split('/')[-3:]).replace('.wav','')

        # create folder if not exists
        new_audio_fldr = os.path.dirname(new_audio_path)

        if not os.path.exists(new_audio_fldr):
            os.makedirs(new_audio_fldr)

        if os.path.exists(audio_filename):
            try:
                if feat == 'melspec':
                    audio_feat = mel_spec(audio_filename)
                elif feat == 'mfcc':
                    audio_feat = feat_mfcc(audio_filename)
                elif feat == 'mfcc_del':
                    audio_feat = feat_mfcc_delta(audio_filename)
                elif feat == 'mfcc_del_del':
                    audio_feat = feat_mfcc_delta_delta(audio_filename)
                np.savez(new_audio_path, audio_feat=audio_feat)
                
            except Exception as e:
                print(f'{audio_filename} has issue')
                
            feat_info.append([os.path.join(feat_sub_path, *audio_filename.split('/')[-3:]).replace('.wav','.npz'), line['text'], line['label']])    
        else:
            print(f"File not found: {audio_filename}")

feat_info_df = pd.DataFrame(feat_info, columns=['audio_path', 'text', 'label'])
feat_info_df.to_csv(feat_pth + f'{set}_{feat}_set_0.csv', sep=',', index=False)
print("Completed!!! ")