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
    mel_feat = logfbank(sig,rate, nfilt=40)

    return mel_feat

def extract_features():
    pass

def compute_delta(wav_file):
    (rate,sig) = wav.read(wav_file)
    mfcc_feat = mfcc(sig,rate)
    mfcc_delta = delta(mfcc_feat, 1)
    delta_feat = np.concatenate((mfcc_feat, mfcc_delta), axis=1)

    return delta_feat


if __name__ == '__main__':
    data_pth = '/home/anuprabha/Desktop/anu_donot_touch/data'
    feat_pth ='/home/anuprabha/Desktop/anu_donot_touch/feats'
    feat_sub_path = 'uaspeech/ft_ctrl_4spkr/sil_trimmed/melspec/train'
    metadata_path = '/home/anuprabha/Desktop/anu_donot_touch/data/UASpeech/UASpeech_noisereduce/FT_ctrl_4spkr/sil_trimmed/train.csv'

    metadata = pd.read_csv(metadata_path)

    feat_info = []

    for index,line  in tqdm(metadata.iterrows()):
        audio_filename = os.path.join(data_pth,line['audio_path'])#.replace('.wav','')
        new_audio_path = os.path.join(os.path.join(feat_pth, feat_sub_path), *audio_filename.split('/')[-3:]).replace('.wav','')

        # create folder if not exists
        new_audio_fldr = os.path.dirname(new_audio_path)
        if not os.path.exists(new_audio_fldr):
            os.makedirs(new_audio_fldr)

        if os.path.exists(audio_filename):
            try:
                audio_feat = mel_spec(audio_filename)
                # audio_feat = compute_delta(audio_filename)
                np.savez(new_audio_path, audio_feat=audio_feat)
                
            except Exception as e:
                print(f'{audio_filename} has issue')
                
            feat_info.append([os.path.join(feat_sub_path, *audio_filename.split('/')[-3:]).replace('.wav','.npz'), line['text'], line['label']])    
        else:
            print(f"File not found: {audio_filename}")

feat_info_df = pd.DataFrame(feat_info, columns=['audio_path', 'text', 'label'])
feat_info_df.to_csv(os.path.join(feat_pth, feat_sub_path) + '_feats.csv', sep=',', index=False)
print("Completed!!! ")